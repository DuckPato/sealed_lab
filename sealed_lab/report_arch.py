"""Archetypes and winning deck shape for the Set report (phase 2).  Read-only on the corpus and the model.

    archetypes_section(...) -> dict   level 1 = main colour pair ("UB+r" / "UB+r/g" when a colour is splashed in >= SPLASH_COLOR_X of its decks), its style
                                       spectrum (main axis, two ends, win-rate difference; see `spectrum_axis`), level 2 = sub-archetypes
                                       found in the decks' card lists, shown ONLY when they replicate on two random halves of the events
    shape_section(...)      -> dict   ranges most decks use (lands, creatures, 2-drops, average mana value, splash cards), and the same among
                                       decks with strong results

How the sub-archetypes are decided (all numbers are constants below):
  1. Each deck of a pair is described by (a) the non-land cards it plays that fit the pair's two colours (splash cards are left out, so the
     pair does not simply split into "with a red splash / without"), weighted so that cards almost every deck of the pair plays count
     little (idf = log((N + 1) / (df + 1))), and (b) its STYLE: shares of cheap (mana value <= 2) / 3 / 4 / 5+ spells, creature share,
     removal count, card-draw count, flier count, land count.  The two blocks get the same total variance, then a PCA keeps EMBED_RANK axes.
  2. k-means with k = 2..K_MAX.  A k is accepted only if it REPLICATES: the events (draft_id) are split at random into two halves, the whole
     pipeline (weights, PCA, k-means) is fitted on each half, and both fits assign ALL the pair's decks; the adjusted Rand index of the two
     assignments must be >= ARI_MIN (median over N_SPLITS random halvings), and every cluster of the fit on all decks must hold
     >= MIN_CLUSTER_SHARE of the pair's decks and >= MIN_CLUSTER_DECKS decks.  Among the accepted k the one with the best silhouette wins (so a clean 2-group structure is not cut into 3); none accepted = one archetype.
     Extra guard (REQUIRE_DISTINCT, ours, not in the brief): decks of a pair vary along smooth axes (spells vs creatures, curve), and
     k-means cuts such a cloud into halves that REPLICATE well without being groups.  So the silhouette of the split must also beat the
     simulated smooth clouds (N_NULL Gaussian samples with the same covariance as the decks' embedding) by DISTINCT_Z standard
     deviations and by at least DISTINCT_GAP.
  3. A sub-archetype is named from its shape against the rest of its pair ("aggro": lower curve, not fewer creatures; "control": higher
     curve with fewer creatures or more removal / card draw; otherwise by its most over-played card) and its 3 most typical cards
     (largest rate x log-lift against the pair, at least SIG_MIN_DECKS decks and a rate of SIG_MIN_RATE).
  4. Win rates: one penalised logistic fit with the model's pilot covariates (skill, play/draw, mulligans, event, time, opponent colours)
     and a dummy per archetype, as in the colour-pair table of `report`; effects are against an average deck, standard errors robust to
     the games of one event.  A sub-archetype also gets its effect against the rest of ITS pair.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from sealed_lab import config, report as R
from sealed_lab.model import archetypes as ARCH
from sealed_lab.model import attributes, design

log = logging.getLogger(__name__)

SPLASH_X = 0.25              # kept for the report's metadata: a splash colour is named in the pair's code from this share of its decks (SPLASH_COLOR_X)
AXIS_MIN_CORR = 0.7          # the style axis is shown only when two independent halves give deck scores that correlate at least this much
K_MAX = 4
MIN_SPLIT_DECKS = 400        # a pair needs this many decks before a split is even tried (2 clusters x MIN_CLUSTER_DECKS, with room)
MIN_CLUSTER_SHARE = 0.10
MIN_CLUSTER_DECKS = 150
ARI_MIN = 0.5
N_SPLITS = 3                 # random halvings tried; the median ARI is used
EMBED_RANK = 10
REQUIRE_DISTINCT = True      # a split must also be sharper than a smooth cloud of decks (see `cluster_pair`); False = replication and size only
N_NULL = 12                  # smooth-cloud simulations per k
DISTINCT_Z, DISTINCT_GAP = 3.0, 0.01   # the split's silhouette must exceed the smooth clouds' mean by this many sd AND this much
SIL_SAMPLE = 1000
MIN_CARD_DECKS = 10          # a card enters the card block when at least this many decks of the pair play it
SIG_N = 3
SIG_MIN_DECKS = 30
SIG_MIN_RATE = 0.20
BEST_N = 5
BEST_MIN_DECKS = 25
BEST_MIN_SHARE = 0.05
MATCHUP_MIN_GAMES = 120      # games of a pair against an opponent pair before that matchup is judged
STRONG_MIN_DECKS = 40        # decks with strong results before their ranges are shown
SHAPE_PCT = (25, 75)
STYLE_NAMES = ["cheap", "mv3", "mv4", "mv5p", "creature_share", "removal", "draw", "flying", "lands"]
SHAPE_METRICS = [("lands", "Lands"), ("creatures", "Creatures"), ("two_drops", "2-drops"), ("avg_mv", "Average mana value"),
                 ("splash_cards", "Splash cards")]
BUILDER_NOTE = "Observed only: the builder values each card the same in every archetype (tested; archetype-specific values did not improve predictions)."
SHAPE_NOTE = ("These are descriptions, not advice: for reference, the builder handles curve and land count through the model, and earlier tests found "
              "that these shape differences do not cost games beyond what the model already accounts for.")


# ---------------------------------------------------------------------------------------------
# decks: card vectors, shape statistics
# ---------------------------------------------------------------------------------------------

def colour_mask(s: str | None) -> int:
    return sum(1 << i for i, c in enumerate(R.COLORS) if c in (s or ""))


@dataclass
class CardVectors:
    """Per-card arrays (cards sorted by card_id) used to turn a deck's copies into shape statistics."""
    mv: np.ndarray
    land: np.ndarray
    creature: np.ndarray            # non-land creatures
    cmask: np.ndarray               # colour bits (WUBRG)
    removal: np.ndarray
    draw: np.ndarray
    flying: np.ndarray
    names: list[str]
    basic: np.ndarray

    @classmethod
    def from_cards(cls, cards: pl.DataFrame) -> "CardVectors":
        c = cards.sort("card_id")
        A, nm = attributes.attribute_matrix(c)
        col = lambda n: A[:, nm.index(n)]
        land = c["is_land"].to_numpy().astype(bool)
        return cls(mv=c["mana_value"].fill_null(0.0).to_numpy().astype(float), land=land,
                   creature=c["is_creature"].fill_null(False).to_numpy().astype(bool) & ~land,
                   cmask=np.array([colour_mask(s) for s in c["colors"].to_list()], dtype=np.int64),
                   removal=col("tag_removal"), draw=col("tag_draw"), flying=col("kw_flying"),
                   names=[str(n).split(" // ")[0] for n in c["name"].to_list()], basic=c["is_basic"].to_numpy().astype(bool))


def deck_shape(X: sp.csr_matrix, cv: CardVectors, main_masks: np.ndarray) -> dict[str, np.ndarray]:
    """Shape statistics of every deck (row of the copies matrix X over ALL cards, lands included)."""
    spell = ~cv.land
    f = lambda v: np.asarray(X @ v.astype(float)).ravel()
    lands, spells = f(cv.land), f(spell)
    sp_ = np.maximum(spells, 1.0)
    mv_sum = f(np.where(spell, cv.mv, 0.0))
    out = dict(lands=lands, spells=spells, creatures=f(cv.creature), two_drops=f(cv.creature & (cv.mv == 2)), avg_mv=mv_sum / sp_,
               cheap=f(spell & (cv.mv <= 2)), mv3=f(spell & (cv.mv == 3)), mv4=f(spell & (cv.mv == 4)), mv5p=f(spell & (cv.mv >= 5)),
               removal=f(cv.removal * spell), draw=f(cv.draw * spell), flying=f(cv.flying * spell))
    off = np.zeros(X.shape[0])
    for m in np.unique(cv.cmask[spell]):                 # spells with a colour that is not one of the deck's main colours
        if m:
            off += f(spell & (cv.cmask == m)) * ((int(m) & ~main_masks) != 0)
    out["splash_cards"] = off
    return out


def style_matrix(sh: dict[str, np.ndarray]) -> np.ndarray:
    sp_ = np.maximum(sh["spells"], 1.0)
    return np.column_stack([sh["cheap"] / sp_, sh["mv3"] / sp_, sh["mv4"] / sp_, sh["mv5p"] / sp_, sh["creatures"] / sp_,
                            sh["removal"], sh["draw"], sh["flying"], sh["lands"]])


# ---------------------------------------------------------------------------------------------
# clustering with replication
# ---------------------------------------------------------------------------------------------

def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings of the same items (1 = same grouping, about 0 = unrelated)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return 1.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    ct = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(ct, (ia, ib), 1.0)
    c2 = lambda x: x * (x - 1.0) / 2.0
    s_ij, s_a, s_b, tot = c2(ct).sum(), c2(ct.sum(1)).sum(), c2(ct.sum(0)).sum(), c2(float(n))
    exp = s_a * s_b / tot
    mx = (s_a + s_b) / 2.0
    return 1.0 if mx == exp else float((s_ij - exp) / (mx - exp))


class Embedder:
    """Weights + PCA fitted on some decks, applicable to any: [idf-weighted card block | standardised style block] -> EMBED_RANK axes."""

    def __init__(self, Xb: sp.csr_matrix, S: np.ndarray, rank: int = EMBED_RANK, min_df: int = MIN_CARD_DECKS):
        n = Xb.shape[0]
        df = np.asarray(Xb.sum(0)).ravel()
        self.cols = np.nonzero((df >= min_df) & (df <= 0.98 * n))[0]
        self.idf = np.log((n + 1.0) / (df[self.cols] + 1.0))
        B = Xb[:, self.cols].multiply(self.idf).toarray() if len(self.cols) else np.zeros((n, 0))
        var_b = float(B.var(0).sum()) if B.shape[1] else 0.0
        self.sc_b = 1.0 / np.sqrt(var_b) if var_b > 0 else 0.0
        self.s_mu = S.mean(0)
        sd = S.std(0)
        self.s_sd = np.where(sd > 1e-9, sd, 1.0)
        self.sc_s = 1.0 / np.sqrt(S.shape[1])
        Z = np.hstack([B * self.sc_b, (S - self.s_mu) / self.s_sd * self.sc_s])
        self.mu = Z.mean(0)
        Zc = Z - self.mu
        w, V = np.linalg.eigh(Zc.T @ Zc / max(n, 1))
        self.V = V[:, np.argsort(-w)[:min(rank, Z.shape[1])]]

    def transform(self, Xb: sp.csr_matrix, S: np.ndarray) -> np.ndarray:
        B = Xb[:, self.cols].multiply(self.idf).toarray() if len(self.cols) else np.zeros((Xb.shape[0], 0))
        Z = np.hstack([B * self.sc_b, (S - self.s_mu) / self.s_sd * self.sc_s])
        return (Z - self.mu) @ self.V


def _assign(E: np.ndarray, cent: np.ndarray) -> np.ndarray:
    return ARCH._sqdist(E, cent).argmin(1)


def _centroids(E: np.ndarray, lab: np.ndarray, k: int) -> np.ndarray:
    return np.array([E[lab == j].mean(0) if (lab == j).any() else E[0] for j in range(k)])


def cluster_pair(Xb: sp.csr_matrix, S: np.ndarray, draft_codes: np.ndarray, seed: int = 0, k_max: int = K_MAX,
                 ari_min: float = ARI_MIN, min_share: float = MIN_CLUSTER_SHARE, min_decks: int = MIN_CLUSTER_DECKS,
                 n_splits: int = N_SPLITS, min_total: int = MIN_SPLIT_DECKS, require_distinct: bool | None = None) -> dict:
    """Decide the sub-archetypes of ONE colour pair.  Xb = binary (decks x cards) of the pair-coloured spells, S = style matrix,
    draft_codes = event of every deck.  Returns dict(k, labels (ordered by cluster size), ari, tried=[per k: ari, min_share, ...])."""
    if require_distinct is None:
        require_distinct = REQUIRE_DISTINCT
    n = Xb.shape[0]
    res: dict = dict(k=1, labels=np.zeros(n, dtype=np.int64), ari=None, silhouette=None, tried=[], n_decks=n)
    if n < min_total:
        res["why"] = f"fewer than {min_total} decks"
        return res
    ev = np.unique(draft_codes)
    emb_all = Embedder(Xb, S)
    E_all = emb_all.transform(Xb, S)
    halves = []
    for r in range(n_splits):
        rng = np.random.default_rng(seed * 1009 + r)
        pick = set(rng.permutation(ev)[: len(ev) // 2].tolist())
        in_a = np.array([c in pick for c in draft_codes.tolist()])
        if in_a.sum() < 50 or (~in_a).sum() < 50:
            continue
        sides = []
        for m in (in_a, ~in_a):
            em = Embedder(Xb[m], S[m])
            sides.append((em, em.transform(Xb[m], S[m]), em.transform(Xb, S)))
        halves.append(sides)
    if not halves:
        res["why"] = "too few events to split in halves"
        return res
    best = None
    for k in range(2, k_max + 1):
        aris = []
        for r, sides in enumerate(halves):
            labs = []
            for j, (em, Etr, Eall) in enumerate(sides):
                lab, _ = ARCH.kmeans(Etr, k, seed=seed * 31 + 7 * k + 3 * r + j, n_init=4)
                labs.append(_assign(Eall, _centroids(Etr, lab, k)))
            aris.append(adjusted_rand(labs[0], labs[1]))
        lab_all, _ = ARCH.kmeans(E_all, k, seed=seed * 31 + 7 * k, n_init=6)
        sizes = np.bincount(lab_all, minlength=k)
        smin_share, smin_n = float(sizes.min() / n), int(sizes.min())
        ari = float(np.median(aris))
        replicates = ari >= ari_min and smin_share >= min_share and smin_n >= min_decks
        sil = ARCH.silhouette(E_all, lab_all, seed=seed, max_n=SIL_SAMPLE)
        null_max, distinct = None, None
        if replicates:
            rng = np.random.default_rng(seed * 77 + k)
            cov = np.cov(E_all.T)
            nulls = []
            for _ in range(N_NULL):
                Z = rng.multivariate_normal(np.zeros(E_all.shape[1]), cov, size=n)
                lz, _i = ARCH.kmeans(Z, k, seed=seed + k, n_init=3)
                nulls.append(ARCH.silhouette(Z, lz, seed=seed, max_n=SIL_SAMPLE))
            null_max = float(np.mean(nulls) + max(DISTINCT_Z * np.std(nulls), DISTINCT_GAP))
            distinct = bool(sil > null_max)
        ok = replicates and (distinct or not require_distinct)
        why = "" if ok else ("clusters do not match across halves" if ari < ari_min else "a cluster is too small" if not replicates
                             else "not sharper than a smooth cloud of decks")
        res["tried"].append(dict(k=k, ari=ari, aris=[float(a) for a in aris], min_share=smin_share, min_decks=smin_n, silhouette=float(sil),
                                 silhouette_needed=null_max, replicates=bool(replicates), distinct=distinct, accepted=bool(ok), why=why))
        if ok and (best is None or sil > best[3]):
            best = (k, lab_all, ari, sil)
    if best is not None:
        k, lab, ari, sil = best
        order = np.argsort(-np.bincount(lab, minlength=k), kind="stable")          # biggest cluster first
        remap = np.empty(k, dtype=np.int64)
        remap[order] = np.arange(k)
        res.update(k=k, labels=remap[lab], ari=ari, silhouette=float(sil))
    return res


# ---------------------------------------------------------------------------------------------
# describing an archetype
# ---------------------------------------------------------------------------------------------

def signature_cards(B: sp.csr_matrix, member: np.ndarray, ref: np.ndarray, cv: CardVectors, n: int = SIG_N) -> list[dict]:
    """Most typical cards of the decks `member` (bool) against `ref` (bool rows to compare with): rate x log-lift, spells only."""
    nm = int(member.sum())
    if nm == 0:
        return []
    cnt = np.asarray(B[member].sum(0)).ravel()
    rate = cnt / nm
    rref = np.asarray(B[ref].sum(0)).ravel() / max(int(ref.sum()), 1)
    score = rate * np.log2((rate + 0.02) / (rref + 0.02))
    ok = (cnt >= SIG_MIN_DECKS) & (rate >= SIG_MIN_RATE) & (score > 0) & ~cv.land
    order = [int(c) for c in np.argsort(-score) if ok[c]][:n]
    return [dict(card_id=c, name=cv.names[c], rate=float(rate[c]), ref_rate=float(rref[c])) for c in order]


def style_words(d: dict) -> tuple[list[str], str]:
    """(plain shape words, style name or "") of a group of decks from its differences with its pair: d has dmv, dcre, drem, ddraw, dfly, dcheap."""
    words = []
    if d["dmv"] <= -0.10:
        words.append("lower curve")
    elif d["dmv"] >= 0.10:
        words.append("higher curve")
    if d["dcre"] >= 0.03:
        words.append("more creatures")
    elif d["dcre"] <= -0.03:
        words.append("fewer creatures")
    if d["drem"] >= 0.4:
        words.append("more removal")
    if d["ddraw"] >= 0.3:
        words.append("more card draw")
    if d["dfly"] >= 0.5:
        words.append("more fliers")
    name = ""
    if d["dmv"] <= -0.10 and d["dcre"] >= 0.0:
        name = "aggro"
    elif d["dmv"] >= 0.10 and (d["dcre"] <= 0.0 or d["drem"] + d["ddraw"] >= 0.3):
        name = "control"
    return words, name


def _med(a: np.ndarray) -> float:
    return float(np.median(a)) if len(a) else float("nan")


def shape_summary(sh: dict[str, np.ndarray], m: np.ndarray, splash: np.ndarray) -> dict:
    """Typical shape (medians) of the decks m: lands, creatures, 2-drops, average mana value, splash cards, share of decks that splash."""
    return dict(lands=_med(sh["lands"][m]), creatures=_med(sh["creatures"][m]), two_drops=_med(sh["two_drops"][m]), avg_mv=_med(sh["avg_mv"][m]),
                splash_cards=_med(sh["splash_cards"][m]), splash_share=float(splash[m].mean()) if m.any() else 0.0)


def effect_sentence(a: dict) -> str:
    if a.get("effect_pp") is None:
        return ""
    if a["badge"] == "weak":
        return "No clear win-rate difference from an average deck."
    return f"{R.signed(a['effect_pp'])} points of win rate against an average deck ({a['badge']})."


# ---------------------------------------------------------------------------------------------
# win rates of the archetypes
# ---------------------------------------------------------------------------------------------

def fit_effects(g: pl.DataFrame, akey: np.ndarray, pair_of: dict[str, str], diffs: list[tuple[str, str]] | None = None) -> dict[str, dict]:
    """Skill-adjusted effect of every archetype key (rows of `g` in the same order as `akey`) against the game-weighted average deck; the keys of
    a pair with several members also get the effect against the rest of their pair.  {key: dict(eff, se, z, in_pair_eff, in_pair_se)}.
    `diffs` = [(key_a, key_b)] adds the difference a - b (same units) under out["__diffs__"]["a|b"]."""
    n = g.height
    diffs = [d for d in (diffs or [])]
    uniq, cnt = np.unique(akey, return_counts=True)
    keys = uniq[np.argsort(-cnt, kind="stable")].tolist()
    if len(keys) < 2 or n < 500:
        return {}
    ref, others = keys[0], keys[1:]
    ix = {k: i for i, k in enumerate(others)}
    w = {k: float(c) / n for k, c in zip(uniq.tolist(), cnt.tolist())}
    rows_i = np.nonzero(akey != ref)[0]
    D = sp.csr_matrix((np.ones(len(rows_i)), (rows_i, [ix[k] for k in akey[rows_i]])), shape=(n, len(others)))
    spl = sp.csr_matrix((g["splash_colors"].str.len_chars() > 0).cast(pl.Float64).to_numpy()[:, None])
    extra = sp.hstack([D, spl], format="csr")
    members: dict[str, list[str]] = {}
    for k in keys:
        members.setdefault(pair_of[k], []).append(k)
    multi = [k for k in keys if len(members[pair_of[k]]) > 1]
    diffs = [(a, b) for a, b in diffs if a in w and b in w]

    def contrasts(k0: int, ncol: int) -> np.ndarray:
        L = np.zeros((len(keys) + len(multi) + len(diffs), ncol))
        for r_, (ka, kb) in enumerate(diffs, start=len(keys) + len(multi)):
            if ka != ref:
                L[r_, k0 + ix[ka]] += 1.0
            if kb != ref:
                L[r_, k0 + ix[kb]] -= 1.0
        for r_, key in enumerate(keys):
            if key != ref:
                L[r_, k0 + ix[key]] += 1.0
            for h in others:
                L[r_, k0 + ix[h]] -= w[h]
        for r_, key in enumerate(multi, start=len(keys)):
            pk = members[pair_of[key]]
            tot = sum(w[h] for h in pk)
            if key != ref:
                L[r_, k0 + ix[key]] += 1.0
            for h in pk:
                if h != ref:
                    L[r_, k0 + ix[h]] -= w[h] / tot
        return L

    coef, enc, _prob, c_rob, _eta = R._fit(g, extra, np.full(len(others) + 1, 10.0), contrasts)
    k0 = enc.k
    b = {ref: 0.0, **{key: float(coef[k0 + ix[key]]) for key in others}}
    mean_b = sum(w[key] * b[key] for key in keys)
    out: dict[str, dict] = {}
    for r_, key in enumerate(keys):
        eff, se = R.PT * (b[key] - mean_b), R.PT * float(c_rob[r_])
        out[key] = dict(eff=eff, se=se, z=eff / se if se > 0 else None)
    for r_, key in enumerate(multi, start=len(keys)):
        pk = members[pair_of[key]]
        tot = sum(w[h] for h in pk)
        eff = R.PT * (b[key] - sum(w[h] * b[h] for h in pk) / tot)
        se = R.PT * float(c_rob[r_])
        out[key].update(in_pair_eff=eff, in_pair_se=se, in_pair_z=eff / se if se > 0 else None)
    if diffs:
        out["__diffs__"] = {}
        for r_, (ka, kb) in enumerate(diffs, start=len(keys) + len(multi)):
            eff, se = R.PT * (b[ka] - b[kb]), R.PT * float(c_rob[r_])
            out["__diffs__"][f"{ka}|{kb}"] = dict(eff=eff, se=se, z=eff / se if se > 0 else None)
    return out


def effect_fields(eff: float, se: float, prefix: str = "") -> dict:
    z = eff / se if se > 0 else None
    return {prefix + "effect_pp": eff, prefix + "se_pp": se, prefix + "z": z, prefix + "badge": R.badge_of(z), prefix + "range": R.range_text(eff, se)}


def matchups(g: pl.DataFrame, pairs: list[str]) -> dict[str, dict]:
    """Best-of-three: for each pair its best and worst matchup against another pair, as the extra win rate over what its usual opponents give,
    adjusted for skill and for opponent colours (the model's opponent covariates).  Only with a badge better than Weak and
    MATCHUP_MIN_GAMES games.  {pair: dict(best=..., worst=...)} (entries may be missing)."""
    n = g.height
    if n < 1500 or "opp_colors" not in g.columns or g["opp_colors"].null_count() > 0.5 * n:
        return {}
    own = np.array([R.pair_group(s) or "" for s in g["main_colors"].to_list()])
    opp = np.array([design.norm_pair(s) for s in g["opp_colors"].fill_null("").to_list()])
    pset = [p for p in pairs if p in design.PAIRS]
    cells = [(p, q) for p in pset for q in design.PAIRS]
    cix = {c: i for i, c in enumerate(cells)}
    ri, ci = [], []
    for i, (p, q) in enumerate(zip(own.tolist(), opp.tolist())):
        j = cix.get((p, q))
        if j is not None:
            ri.append(i)
            ci.append(j)
    if not ri:
        return {}
    M = sp.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(n, len(cells)))
    games = np.asarray(M.sum(0)).ravel()
    rows = [(p, q) for (p, q) in cells if p != q and games[cix[(p, q)]] >= MATCHUP_MIN_GAMES]
    if not rows:
        return {}
    own_cols = {p: i for i, p in enumerate(pset[1:])}                   # own-pair main effects (the contrasts below difference them away)
    r_o = [i for i, p in enumerate(own.tolist()) if p in own_cols]
    D = sp.csr_matrix((np.ones(len(r_o)), (r_o, [own_cols[own[i]] for i in r_o])), shape=(n, len(own_cols)))
    extra = sp.hstack([D, M], format="csr")
    sig = np.concatenate([np.full(D.shape[1], 10.0), np.full(len(cells), 0.4)])
    cache: dict = {}

    def contrasts(k0: int, ncol: int) -> np.ndarray:
        L = np.zeros((len(rows), ncol))
        for r_, (p, q) in enumerate(rows):
            wsum = max(sum(games[cix[(p, q2)]] for q2 in design.PAIRS), 1.0)
            L[r_, k0 + D.shape[1] + cix[(p, q)]] += 1.0
            for q2 in design.PAIRS:
                L[r_, k0 + D.shape[1] + cix[(p, q2)]] -= games[cix[(p, q2)]] / wsum
        cache["L"] = L
        return L

    coef, enc, _prob, c_rob, _eta = R._fit(g, extra, sig, contrasts)
    effs = cache["L"] @ coef
    out: dict[str, dict] = {}
    for r_, (p, q) in enumerate(rows):
        eff = R.PT * float(effs[r_])
        se = R.PT * float(c_rob[r_])
        z = eff / se if se > 0 else None
        item = dict(vs=q, vs_label=R.group_label(q), effect_pp=eff, se_pp=se, z=z, badge=R.badge_of(z), range=R.range_text(eff, se),
                    n_games=int(games[cix[(p, q)]]))
        d = out.setdefault(p, {})
        if item["badge"] != "weak":
            if eff > 0 and eff > d.get("best", {}).get("effect_pp", -1e9):
                d["best"] = item
            if eff < 0 and eff < d.get("worst", {}).get("effect_pp", 1e9):
                d["worst"] = item
    return {p: d for p, d in out.items() if d}


# ---------------------------------------------------------------------------------------------
# style spectrum of a colour pair (the main axis along which its decks vary)
# ---------------------------------------------------------------------------------------------

def spectrum_axis(Xb: sp.csr_matrix, S: np.ndarray, draft_codes: np.ndarray, seed: int = 0, n_splits: int = N_SPLITS,
                  min_total: int = MIN_SPLIT_DECKS) -> dict | None:
    """First principal component of the pair's deck features (same pipeline as the clustering).  Replication: the events are split at random in
    two halves, the axis is fitted on each half, all the pair's decks are scored with both, and the absolute correlation of the two scores
    is taken (median over the halvings).  Returns dict(score (all decks), corr, corrs, ok, half_a (bool per deck of the halving with the
    median correlation)) or None if the pair is too small."""
    n = Xb.shape[0]
    if n < min_total:
        return None
    score = Embedder(Xb, S, rank=1).transform(Xb, S)[:, 0]
    ev = np.unique(draft_codes)
    corrs, masks = [], []
    for r in range(n_splits):
        rng = np.random.default_rng(seed * 1009 + 17 + r)
        pick = rng.permutation(ev)[: len(ev) // 2]
        in_a = np.isin(draft_codes, pick)
        if in_a.sum() < 50 or (~in_a).sum() < 50:
            continue
        sc = [Embedder(Xb[m], S[m], rank=1).transform(Xb, S)[:, 0] for m in (in_a, ~in_a)]
        c = np.corrcoef(sc[0], sc[1])[0, 1] if sc[0].std() > 0 and sc[1].std() > 0 else 0.0
        corrs.append(abs(float(c)) if np.isfinite(c) else 0.0)
        masks.append(in_a)
    if not corrs:
        return None
    med = float(np.median(corrs))
    j = int(np.argmin([abs(c - med) for c in corrs]))
    return dict(score=score, corr=med, corrs=corrs, ok=bool(med >= AXIS_MIN_CORR), half_a=masks[j])


SPECTRUM_FEATURES = [          # key, threshold on the difference between the ends, (label token, phrase) of the end with more, same for the end with less
    ("avg_mv", 0.15, ("high curve", "higher curve"), ("low curve", "lower curve")),
    ("cre_share", 0.04, ("creatures", "more creatures"), None),
    ("removal", 0.5, ("removal", "more removal"), None),
    ("draw", 0.4, ("card draw", "more card draw"), None),
    ("flying", 0.7, ("fliers", "more fliers"), None),
    ("two_drops", 0.8, ("2-drops", "more 2-drops"), None),
]
SPECTRUM_MAX_WORDS = 3


def _join_words(w: list[str]) -> str:
    return ", ".join(w[:-1]) + " and " + w[-1] if len(w) > 1 else (w[0] if w else "")


def describe_ends(sh: dict[str, np.ndarray], lo: np.ndarray, hi: np.ndarray, allm: np.ndarray) -> tuple[dict, dict, dict]:
    """Plain-word description of the two ends (bool masks lo / hi over the decks, `allm` = the pair's decks): (end_lo, end_hi, differences);
    an end is dict(label, words, description).  Only differences between the ends that clear a threshold are named, strongest first (at most
    SPECTRUM_MAX_WORDS per end, the label uses the first two); the end with more of something names it, and the curve is named on both ends."""
    sp_ = np.maximum(sh["spells"], 1.0)
    vals = dict(avg_mv=sh["avg_mv"], cre_share=sh["creatures"] / sp_, removal=sh["removal"], draw=sh["draw"], flying=sh["flying"], two_drops=sh["two_drops"])
    diffs, found = {}, []
    for key, thr, more, less in SPECTRUM_FEATURES:
        v = vals[key]
        d = float(v[hi].mean() - v[lo].mean())
        diffs[key] = d
        if abs(d) >= thr:
            sd = float(v[allm].std()) or 1.0
            found.append((abs(d) / sd, d > 0, more, less))
    found.sort(key=lambda t: -t[0])
    ends = {False: dict(tokens=[], words=[]), True: dict(tokens=[], words=[])}
    for _st, up, more, less in found:
        ends[up]["tokens"].append(more[0])
        ends[up]["words"].append(more[1])
        if less:
            ends[not up]["tokens"].append(less[0])
            ends[not up]["words"].append(less[1])
    out = [dict(label=" / ".join(ends[u]["tokens"][:2]), words=ends[u]["words"][:SPECTRUM_MAX_WORDS],
                description=_join_words(ends[u]["words"][:SPECTRUM_MAX_WORDS])) for u in (False, True)]
    return out[0], out[1], diffs


def pair_win_difference(gp: pl.DataFrame, half_a_events) -> dict:
    """Within ONE colour pair: skill-adjusted win-rate difference (points) between the top and the bottom third of the style axis, fitted separately
    on the games of each half of the events (draft_id in `half_a_events` / not).  gp needs the column `lvl` (0 = bottom, 1 = middle, 2 = top third).
    Returns dict(halves=[dict(eff, se, z) | None, ...])."""
    def one(h: pl.DataFrame) -> dict | None:
        lv = h["lvl"].to_numpy()
        if h.height < 400 or any((lv == v).sum() < 50 for v in (0, 2)):
            return None
        spl = (h["splash_colors"].str.len_chars() > 0).cast(pl.Float64).to_numpy()
        D = sp.csr_matrix(np.column_stack([lv == 0, lv == 2, spl]).astype(float))

        def contrasts(k0: int, ncol: int) -> np.ndarray:
            L = np.zeros((1, ncol))
            L[0, k0 + 1], L[0, k0] = 1.0, -1.0
            return L

        coef, enc, _prob, c_rob, _eta = R._fit(h, D, np.full(3, 10.0), contrasts)
        eff, se = R.PT * float(coef[enc.k + 1] - coef[enc.k]), R.PT * float(c_rob[0])
        return dict(eff=eff, se=se, z=eff / se if se > 0 else None)

    inA = gp["draft_id"].is_in(list(half_a_events)).to_numpy()
    return dict(halves=[one(gp.filter(pl.Series(m))) for m in (inA, ~inA)])


def spectrum_verdict(full: dict | None, halves: list) -> dict:
    """Is the win-rate difference between the ends a finding?  Badge Likely or Strong on all the data AND the same sign when fitted in both halves."""
    if not full:
        return dict(finding=False, replicated=False)
    sg = [float(np.sign(h["eff"])) if h else 0.0 for h in halves]
    rep = len(sg) == 2 and sg[0] != 0 and sg[0] == sg[1] == float(np.sign(full["eff"]))
    return dict(finding=bool(R.badge_of(full["z"]) in ("likely", "strong") and rep), replicated=bool(rep))


def spectrum_texts(key: str, a: dict, b: dict, win: dict) -> list[str]:
    """Sentences of the spectrum: the two ends, then the win-rate sentence.  a = the lower-curve end, b = the other; win['effect_pp'] = b - a."""
    da, db = a["description"] or a["label"], b["description"] or b["label"]
    out = [f"{key} decks range from {da} to {db} (a spectrum, not separate archetypes)."]
    if win.get("finding"):
        better, worse = (b, a) if win["effect_pp"] > 0 else (a, b)
        out.append(f"Decks at the {better['label']} end win {abs(win['effect_pp']):.1f} points more often than decks at the {worse['label']} end "
                   f"({win['badge']}; the same direction in both halves of the data).")
    else:
        out.append("No clear difference in win rate between the two styles.")
    return out


SPLASH_COLOR_X = 0.25        # a splash colour is named when at least this share of the pair's decks splash it


def splash_colors_of(key: str, counts: dict, n: int, x: float | None = None) -> list[str]:
    """Colours (not in the pair) splashed by at least `x` of the pair's n decks, in WUBRG order."""
    x = SPLASH_COLOR_X if x is None else x
    return [c for c in R.COLORS if c not in key and n > 0 and counts.get(c, 0) / n >= x]


# ---------------------------------------------------------------------------------------------
# the archetype section
# ---------------------------------------------------------------------------------------------

def code_of(key: str, splash_cols: list[str]) -> str:
    """"UB", or "UB+r" / "UB+r/g" when those colours are splashed by at least SPLASH_COLOR_X of the pair's decks; "UB+x" when three
    or more are (it splashes whatever the pool offers, and "UB+w/r/g" reads as noise)."""
    if len(splash_cols) > 2:
        return key + "+x"
    return key + ("+" + "/".join(c.lower() for c in splash_cols) if splash_cols else "")


def deck_table(set_code: str, g: pl.DataFrame, cards: pl.DataFrame) -> tuple[pl.DataFrame, sp.csr_matrix]:
    """One row per deck (draft_id, deck_key) of the games `g`, and the (decks x cards) copies matrix."""
    decks = (g.group_by("draft_id", "deck_key", maintain_order=False)
             .agg(pl.col("main_colors").first(), pl.col("splash_colors").first(), pl.len().alias("games"), pl.col("won").cast(pl.Int32).sum().alias("wins"))
             .sort("draft_id", "deck_key")
             .with_columns(pl.col("main_colors").map_elements(R.pair_group, return_dtype=pl.String).alias("grp"),
                           pl.col("main_colors").map_elements(colour_mask, return_dtype=pl.Int64).alias("mmask"))
             .filter(pl.col("grp").is_not_null())
             .with_row_index("row"))
    cd = config.corpus_dir(set_code)
    dc = (pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"])
          .join(decks.select("row", "deck_key"), on="deck_key"))
    n_cards = int(cards["card_id"].max()) + 1
    X = sp.csr_matrix((dc["n"].to_numpy().astype(float), (dc["row"].to_numpy().astype(np.int64), dc["card_id"].to_numpy().astype(np.int64))),
                      shape=(decks.height, n_cards))
    return decks, X


def _pair_binary(X: sp.csr_matrix, cv: CardVectors, pair_mask: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """(binary copies, card ids) of the non-land cards whose colours fit inside the pair."""
    fits = ~cv.land & ((cv.cmask & ~pair_mask) == 0)
    return (X[:, np.nonzero(fits)[0]] > 0).astype(float).tocsr(), np.nonzero(fits)[0]


def build_archetypes(decks: pl.DataFrame, X: sp.csr_matrix, cv: CardVectors, seed: int = 0, progress=None) -> dict:
    """Decide the archetypes of every colour pair.  Returns dict(akey per deck (array), pairs={pair: cluster_pair result + labelled infos})."""
    sh = deck_shape(X, cv, decks["mmask"].to_numpy())
    S = style_matrix(sh)
    grp = decks["grp"].to_numpy()
    codes = np.unique(decks["draft_id"].to_numpy(), return_inverse=True)[1]
    akey = grp.astype(object).copy()
    out: dict = dict(shape=sh, pairs={})
    for pi, p in enumerate(design.PAIRS):
        m = grp == p
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        Xb, _cols = _pair_binary(X[idx], cv, colour_mask(p))
        res = cluster_pair(Xb, S[idx], codes[idx], seed=seed + pi)
        res["idx"] = idx
        ax = spectrum_axis(Xb, S[idx], codes[idx], seed=seed + pi)
        if ax is not None:                                                  # orient: the higher score = the higher average mana value
            c = np.corrcoef(ax["score"], sh["avg_mv"][idx])[0, 1] if ax["score"].std() > 0 and sh["avg_mv"][idx].std() > 0 else 1.0
            if c < 0:
                ax["score"] = -ax["score"]
        res["axis"] = ax
        out["pairs"][p] = res
        if res["k"] > 1:
            for j in range(res["k"]):
                akey[idx[res["labels"] == j]] = f"{p}#{j + 1}"
    out["akey"] = akey.astype(str)
    return out


def archetypes_section(set_code: str, fmt: str, g: pl.DataFrame, model, pairs_sec: dict, card_rows: list[dict] | None = None, progress=None) -> tuple[dict, dict]:
    """(archetypes section, deck-shape section) for the games `g` (one format) given the colour-pair rows of `report.pairs_section`."""
    R.check_corpus_matches(set_code, model)
    cards = model.cards.sort("card_id")
    cv = CardVectors.from_cards(cards)
    decks, X = deck_table(set_code, g, cards)
    if progress:
        progress(0.1, "Sorting decks into archetypes")
    bld = build_archetypes(decks, X, cv)
    sh, akey = bld["shape"], bld["akey"]
    grp = decks["grp"].to_numpy()
    n_decks = decks.height
    splash_deck = (decks["splash_colors"].str.len_chars() > 0).to_numpy()
    strong = strong_mask(g, decks, fmt)
    # --- win rates
    if progress:
        progress(0.4, "Archetype win rates (adjusted for pilot skill)")
    gk = g.join(decks.select("draft_id", "deck_key", pl.Series("akey", akey), pl.Series("grp2", grp)), on=["draft_id", "deck_key"], how="left")
    ok_rows = gk["akey"].is_not_null().to_numpy()
    g2 = gk.filter(pl.Series(ok_rows))
    pair_of = {k: str(v) for k, v in zip(akey.tolist(), grp.tolist())}
    eff = fit_effects(g2, g2["akey"].to_numpy(), pair_of)
    # --- opponent matchups (best of three only)
    mu: dict = {}
    if fmt == "bo3":
        if progress:
            progress(0.6, "Matchups between colour pairs")
        try:
            mu = matchups(g2, [r["key"] for r in pairs_sec.get("rows", []) if r["key"] in design.PAIRS])
        except Exception:                                                                        # a failed matchup fit must not lose the section
            log.info("matchups failed", exc_info=True)
    prow = {r["key"]: r for r in pairs_sec.get("rows", [])}
    spec_fit = spectrum_fits(g, decks, bld, prow, progress)
    B_all = (X > 0).tocsr()
    vals = {r["card_id"]: r for r in (card_rows or [])}
    # splash colours: the most common splash colour among the splashing decks of the pair
    spl = decks["splash_colors"].to_numpy()
    sec: dict = dict(note=BUILDER_NOTE, splash_x=SPLASH_COLOR_X, axis_min_corr=AXIS_MIN_CORR, pairs=[], n_decks=n_decks,
                     params=dict(ari_min=ARI_MIN, min_cluster_share=MIN_CLUSTER_SHARE, min_cluster_decks=MIN_CLUSTER_DECKS, k_max=K_MAX,
                                 n_splits=N_SPLITS))
    for p in design.PAIRS:
        pr = prow.get(p)
        res = bld["pairs"].get(p)
        if pr is None or res is None or pr["hidden"]:
            continue
        idx = res["idx"]
        mp = np.zeros(n_decks, dtype=bool)
        mp[idx] = True
        splashers = mp & splash_deck
        counts = {c: sum(1 for s in spl[splashers] if c in s) for c in R.COLORS if c not in p}
        top_c = max(counts, key=counts.get) if splashers.any() else ""
        splash_share = float(splashers.sum() / max(mp.sum(), 1))
        splash_cols = splash_colors_of(p, counts, int(mp.sum()))
        pair = dict(key=p, label=R.group_label(p), code=code_of(p, splash_cols), splashy=bool(splash_cols), splash_cols=splash_cols,
                    splash_col_shares={c: counts[c] / max(int(mp.sum()), 1) for c in splash_cols},
                    splash_share=splash_share, splash_top=top_c, splash_top_label=R.COLOR_NAMES.get(top_c, ""),
                    splash_top_share=counts.get(top_c, 0) / max(int(splashers.sum()), 1) if top_c else 0.0,
                    n_decks=int(mp.sum()), share=pr["share"], **{k: pr[k] for k in ("effect_pp", "se_pp", "z", "badge", "range")})
        pair["signature"] = signature_cards(B_all, mp, ~mp, cv)
        pair["shape"] = shape_summary(sh, mp, splash_deck)
        pair["best_cards"] = best_cards(B_all, mp, vals, cv)
        pair["matchups"] = mu.get(p, {})
        pair["split"] = dict(k=res["k"], ari=res["ari"], silhouette=res.get("silhouette"), tried=res["tried"], why=res.get("why", ""))
        pair["subs"] = []
        pair["spectrum"] = spectrum_block(p, res, spec_fit.get(p), sh, mp, B_all, cv)
        if res["k"] > 1:
            ref_style = dict(avg_mv=float(sh["avg_mv"][mp].mean()), cre=float((sh["creatures"][mp] / np.maximum(sh["spells"][mp], 1)).mean()),
                             rem=float(sh["removal"][mp].mean()), draw=float(sh["draw"][mp].mean()), fly=float(sh["flying"][mp].mean()),
                             cheap=float((sh["cheap"][mp] / np.maximum(sh["spells"][mp], 1)).mean()))
            names_used: set[str] = set()
            for j in range(res["k"]):
                key = f"{p}#{j + 1}"
                mj = akey == key
                d = dict(dmv=float(sh["avg_mv"][mj].mean()) - ref_style["avg_mv"],
                         dcre=float((sh["creatures"][mj] / np.maximum(sh["spells"][mj], 1)).mean()) - ref_style["cre"],
                         drem=float(sh["removal"][mj].mean()) - ref_style["rem"], ddraw=float(sh["draw"][mj].mean()) - ref_style["draw"],
                         dfly=float(sh["flying"][mj].mean()) - ref_style["fly"],
                         dcheap=float((sh["cheap"][mj] / np.maximum(sh["spells"][mj], 1)).mean()) - ref_style["cheap"])
                words, style = style_words(d)
                sig = signature_cards(B_all, mj, mp & ~mj, cv)
                name = f"{p} {style}" if style else (f"{p} built around {sig[0]['name']}" if sig else f"{p} group {j + 1}")
                if name in names_used:
                    name = f"{name} ({sig[0]['name']})" if sig else f"{name} {j + 1}"
                names_used.add(name)
                e = eff.get(key, {})
                sub = dict(key=key, name=name, style=style, words=words, n_decks=int(mj.sum()), share_of_pair=float(mj.sum() / max(mp.sum(), 1)),
                           share=float(mj.sum() / max(n_decks, 1)), signature=sig, shape=shape_summary(sh, mj, splash_deck),
                           best_cards=best_cards(B_all, mj, vals, cv), style_diff=d)
                if e:
                    sub.update(effect_fields(e["eff"], e["se"]))
                    if "in_pair_eff" in e:
                        sub.update(effect_fields(e["in_pair_eff"], e["in_pair_se"], "vs_pair_"))
                sub["text"] = sub_text(sub, pair)
                pair["subs"].append(sub)
        pair["text"] = pair_text(pair)
        sec["pairs"].append(pair)
    def by_win(a: dict) -> tuple:                         # best-winning first (owner); no estimate last, then by size
        return (a.get("effect_pp") is None, -(a.get("effect_pp") or 0.0), -a["n_decks"])
    for a in sec["pairs"]:
        a["subs"].sort(key=by_win)
    sec["pairs"].sort(key=by_win)
    sec["n_with_subs"] = sum(1 for a in sec["pairs"] if a["subs"])
    sec["n_with_spectrum"] = sum(1 for a in sec["pairs"] if a["spectrum"].get("shown"))
    # --- deck shape
    shape = shape_section(decks, sh, strong, fmt, splash_deck)
    try:
        from sealed_lab import report_detail

        shape["details"] = report_detail.shape_details(g, decks, sh)
    except Exception:                                                                            # the details must never lose the section
        log.info("shape details failed", exc_info=True)
    return sec, shape


def spectrum_fits(g: pl.DataFrame, decks: pl.DataFrame, bld: dict, prow: dict, progress=None) -> dict[str, dict]:
    """Win-rate side of the style spectra: for every visible pair whose axis replicates, cut its decks in thirds along the axis and fit
    (a) all games at once (effects against an average deck, top - bottom difference with a clustered SE) and (b) each pair separately in the
    two halves of the events (same-sign rule).  {pair: dict(lo_key, hi_key, lo_eff, hi_eff, diff, halves, verdict)}."""
    grp = decks["grp"].to_numpy()
    akey = grp.astype(object).copy()
    lvl = np.full(len(grp), -1, dtype=np.int64)
    info: dict[str, dict] = {}
    for p, res in bld["pairs"].items():
        ax = res.get("axis")
        pr = prow.get(p)
        if ax is None or not ax["ok"] or pr is None or pr["hidden"]:
            continue
        idx, sc = res["idx"], ax["score"]
        q1, q2 = np.quantile(sc, [1 / 3, 2 / 3])
        lv = np.where(sc <= q1, 0, np.where(sc >= q2, 2, 1))
        for v, nm in ((0, "lo"), (1, "mid"), (2, "hi")):
            akey[idx[lv == v]] = f"{p}~{nm}"
        lvl[idx] = lv
        half_ids = set(decks["draft_id"].to_numpy()[idx][ax["half_a"]].tolist())
        info[p] = dict(lo_key=f"{p}~lo", hi_key=f"{p}~hi", half_ids=half_ids)
    if not info:
        return {}
    if progress:
        progress(0.7, "Win rate along each pair's style axis")
    pair_of = {str(k): str(v) for k, v in zip(akey.tolist(), grp.tolist())}
    gk = g.join(decks.select("draft_id", "deck_key", pl.Series("akey", akey.astype(str)), pl.Series("grp2", grp), pl.Series("lvl", lvl)),
                on=["draft_id", "deck_key"], how="inner")
    eff = fit_effects(gk, gk["akey"].to_numpy(), pair_of, diffs=[(v["hi_key"], v["lo_key"]) for v in info.values()])
    out: dict[str, dict] = {}
    for p, v in info.items():
        d = eff.get("__diffs__", {}).get(f"{v['hi_key']}|{v['lo_key']}")
        if d is None or v["lo_key"] not in eff or v["hi_key"] not in eff:
            continue
        gp = gk.filter(pl.col("grp2") == p)
        gp = gp.filter(pl.col("lvl") >= 0)
        halves = pair_win_difference(gp, v["half_ids"])["halves"]
        out[p] = dict(lo_eff=eff[v["lo_key"]], hi_eff=eff[v["hi_key"]], diff=d, halves=halves, verdict=spectrum_verdict(d, halves))
    return out


def spectrum_block(p: str, res: dict, fit: dict | None, sh: dict, mp: np.ndarray, B_all: sp.csr_matrix, cv: CardVectors) -> dict:
    """The 'spectrum' entry of a pair: shown=False (with the axis correlation) or the two ends, their signature cards and the win-rate finding."""
    ax = res.get("axis")
    if ax is None:
        return dict(shown=False, why="too few decks")
    base = dict(corr=ax["corr"], corrs=ax["corrs"], min_corr=AXIS_MIN_CORR)
    if not ax["ok"]:
        return dict(shown=False, why="the main style axis does not replicate", **base)
    if fit is None:
        return dict(shown=False, why="no win-rate fit", **base)
    idx, sc = res["idx"], ax["score"]
    n_all = len(mp)
    q1, q2 = np.quantile(sc, [1 / 3, 2 / 3])
    lo, hi = np.zeros(n_all, dtype=bool), np.zeros(n_all, dtype=bool)
    lo[idx[sc <= q1]] = True
    hi[idx[sc >= q2]] = True
    a, b, diffs = describe_ends(sh, lo, hi, mp)
    ends = []
    for e, m, other, fe in ((a, lo, hi, fit["lo_eff"]), (b, hi, lo, fit["hi_eff"])):
        sig = signature_cards(B_all, m, other, cv)
        fallback = f"built around {sig[0]['name']}" if sig else ("one end" if m is lo else "the other end")
        label = e["label"] or fallback
        ends.append(dict(label=label, words=e["words"], description=e["description"] or label, n_decks=int(m.sum()), signature=sig,
                         avg_mv=float(sh["avg_mv"][m].mean()), **effect_fields(fe["eff"], fe["se"])))
    d, v = fit["diff"], fit["verdict"]
    win = dict(**effect_fields(d["eff"], d["se"]), finding=v["finding"], replicated=v["replicated"],
               halves=[None if h is None else round(h["eff"], 2) for h in fit["halves"]])
    text = spectrum_texts(p, ends[0], ends[1], win)
    return dict(shown=True, **base, a=ends[0], b=ends[1], win=win, differences=diffs, text=text)


def best_cards(B: sp.csr_matrix, member: np.ndarray, vals: dict[int, dict], cv: CardVectors) -> list[dict]:
    """Cards with the highest model value among those the decks `member` play (>= BEST_MIN_SHARE of them, >= BEST_MIN_DECKS)."""
    nm = int(member.sum())
    if nm == 0 or not vals:
        return []
    cnt = np.asarray(B[member].sum(0)).ravel()
    out = []
    for cid, r in vals.items():
        if cid < len(cnt) and cnt[cid] >= max(BEST_MIN_DECKS, BEST_MIN_SHARE * nm) and r["value_pp"] > 0 and not cv.basic[cid]:
            share = float(cnt[cid] / nm)
            out.append(dict(card_id=int(cid), name=r["name"], rarity=r["rarity"], value_pp=r["value_pp"], z=r["z"], badge=r["badge"], range=r["range"],
                            thin=r.get("thin", False), play_rate=share,
                            reason=f"In {share * 100:.0f}% of these decks; {R.signed(r['value_pp'])} points of win rate per copy against the average card of its colour."))
    out.sort(key=lambda r: -r["value_pp"])
    return out[:BEST_N]


def sub_text(sub: dict, pair: dict) -> str:
    s = f"{sub['share_of_pair'] * 100:.0f}% of {pair['key']} decks"
    if sub.get("words"):
        s += f": {', '.join(sub['words'])}"
    s += "."
    if sub.get("vs_pair_badge") and sub["vs_pair_badge"] != "weak":
        better = "better" if sub["vs_pair_effect_pp"] > 0 else "worse"
        s += f" Does {better} than the other {pair['key']} decks by {abs(sub['vs_pair_effect_pp']):.1f} points ({sub['vs_pair_badge']})."
    elif sub.get("vs_pair_badge"):
        s += " No clear win-rate difference from the other " + pair["key"] + " decks."
    return s


def pair_text(pair: dict) -> list[str]:
    out = []
    if pair["splashy"]:
        parts = [f"{R.COLOR_NAMES.get(c, c).lower()} ({pair['splash_col_shares'][c] * 100:.0f}%)" for c in pair["splash_cols"]]
        out.append(f"{pair['label']} decks often splash {' and '.join(parts)}, so it is written {pair['code']}.")
    elif pair["splash_share"] > 0:
        tail = f", most often {pair['splash_top_label']}" if pair["splash_top_label"] else ""
        out.append(f"{pair['splash_share'] * 100:.0f}% of {pair['label']} decks splash a third colour{tail}.")
    if pair["subs"]:
        out.append(f"Two or more styles replicate in independent halves of the data: {', '.join(s['name'] for s in pair['subs'])}.")
    elif pair["split"].get("why"):
        out.append("Too few decks to look for separate styles inside this pair.")
    else:
        out.append("No separate sub-archetypes: the decks of this pair vary smoothly and do not fall into clearly different groups "
                   "that show up the same way in two independent halves of the data.")
    m = pair.get("matchups") or {}
    if m.get("best"):
        out.append(f"Best matchup: against {m['best']['vs_label']} ({R.signed(m['best']['effect_pp'])} points above usual, {m['best']['badge']}).")
    if m.get("worst"):
        out.append(f"Worst matchup: against {m['worst']['vs_label']} ({R.signed(m['worst']['effect_pp'])} points, {m['worst']['badge']}).")
    return out


# ---------------------------------------------------------------------------------------------
# winning deck shape
# ---------------------------------------------------------------------------------------------

def strong_mask(g: pl.DataFrame, decks: pl.DataFrame, fmt: str) -> np.ndarray:
    """Decks with strong results: 5 or more game wins in Best of 1; 3 or more match wins in Best of 3."""
    if fmt == "bo1":
        return decks["wins"].to_numpy() >= 5
    mw = (g.group_by("draft_id", "deck_key", "match_number").agg((pl.col("won").cast(pl.Int32).sum() * 2 > pl.len()).alias("mw"))
          .group_by("draft_id", "deck_key").agg(pl.col("mw").sum().alias("match_wins")))
    j = decks.select("draft_id", "deck_key").join(mw, on=["draft_id", "deck_key"], how="left")
    return j["match_wins"].fill_null(0).to_numpy() >= 3


def range_of(v: np.ndarray) -> dict | None:
    if len(v) == 0:
        return None
    lo, hi = np.percentile(v, SHAPE_PCT)
    return dict(lo=float(lo), med=float(np.median(v)), hi=float(hi))


def shape_rows(sh: dict[str, np.ndarray], m: np.ndarray, strong: np.ndarray) -> dict:
    """Ranges (25th - 75th percentile) of each metric among the decks m, and among the strong-result decks of m."""
    ms = m & strong
    row: dict = dict(n_decks=int(m.sum()), n_strong=int(ms.sum()), metrics={})
    for key, _label in SHAPE_METRICS:
        row["metrics"][key] = dict(all=range_of(sh[key][m]), strong=range_of(sh[key][ms]) if ms.sum() >= STRONG_MIN_DECKS else None)
    return row


def shape_section(decks: pl.DataFrame, sh: dict[str, np.ndarray], strong: np.ndarray, fmt: str, splash_deck: np.ndarray) -> dict:
    grp = decks["grp"].to_numpy()
    rows = [dict(key="all", label="All decks", **shape_rows(sh, np.ones(len(grp), dtype=bool), strong))]
    for p in design.PAIRS:
        m = grp == p
        if m.sum() >= R.MIN_PAIR_DECKS:
            rows.append(dict(key=p, label=R.group_label(p), **shape_rows(sh, m, strong)))
    sdef = "won 5 or more games" if fmt == "bo1" else "won 3 or more matches"
    return dict(rows=rows, metrics=[dict(key=k, label=l) for k, l in SHAPE_METRICS], strong_def=sdef, strong_min_decks=STRONG_MIN_DECKS,
                percentiles=list(SHAPE_PCT), note=SHAPE_NOTE,
                text=[f"Most decks (the middle half) play {_fmt_range(rows[0]['metrics']['lands']['all'], 0)} lands, "
                      f"{_fmt_range(rows[0]['metrics']['creatures']['all'], 0)} creatures and {_fmt_range(rows[0]['metrics']['two_drops']['all'], 0)} two-drops.",
                      f"\"Strong results\" means the deck {sdef}. Skill is not adjusted here, so this shows what such decks look like, not what causes wins."])


def _fmt_range(r: dict | None, digits: int) -> str:
    if not r:
        return "?"
    lo, hi = round(r["lo"], digits), round(r["hi"], digits)
    f = (lambda x: f"{x:.0f}") if digits == 0 else (lambda x: f"{x:.{digits}f}")
    return f(lo) if lo == hi else f"{f(lo)}-{f(hi)}"


# ---------------------------------------------------------------------------------------------
# colour-pair table for any games (used for the draft comparison)
# ---------------------------------------------------------------------------------------------

def pair_table(g: pl.DataFrame) -> dict[str, dict]:
    """{pair: share, effect_pp, se_pp, z, badge} of the ten colour pairs from `report.pairs_section` on any games."""
    sec = R.pairs_section(g)
    return {r["key"]: r for r in sec.get("rows", []) if r["key"] in design.PAIRS and not r["hidden"]}
