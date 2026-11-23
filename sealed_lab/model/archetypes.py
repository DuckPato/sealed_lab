"""Deck archetypes: clusters of real decks by composition.

Pipeline (all numpy / scipy, deterministic):
  1. TF-IDF matrix of the decks (rows) x spells (columns; copies as term frequency, idf = log((N+1)/(df+1)) + 1), rows L2-normalised;
  2. truncated SVD (PCA of the centred matrix, `RANK` axes) -> a low-noise embedding of every deck;
  3. within each colour group (the 10 two-colour pairs, plus 'other' = mono / 3+ colours) k-means (k-means++ start, several
     restarts, fixed seed) for k = 2..K_MAX; k is chosen by the mean silhouette on a subsample and k > 1 is only accepted when
     the best silhouette reaches SIL_MIN and every cluster holds >= MIN_SHARE of the group; otherwise the colour group is one
     archetype;
  4. an archetype's centroid is the mean TF-IDF vector of its decks (L2-normalised); decks (corpus or new) are assigned to
     the centroid with the highest cosine similarity, which is what `Archetypes.assign` does for a user's deck;
  5. label = colour group + its two most distinctive frequent cards; observed game win rates per format (raw, shrunk towards
     the format mean, cluster-bootstrap CI over events) attached from the games table.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from sealed_lab.model import design

log = logging.getLogger(__name__)

RANK = 16
K_MAX = 5
SIL_MIN = 0.10
MIN_SHARE = 0.08
MIN_GROUP_DECKS = 200
SHRINK_GAMES = 100.0
N_BOOT = 200
SIL_SAMPLE = 2000


# ---------------------------------------------------------------------------------------------
# small numpy clustering helpers
# ---------------------------------------------------------------------------------------------

def _sqdist(X: np.ndarray, C: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
    """Squared euclidean distances (rows of X x rows of C); `x2` = (X ** 2).sum(1) when the caller has it (same arithmetic)."""
    if x2 is None:
        x2 = (X ** 2).sum(1)
    return np.maximum(x2[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0.0)


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 4, iters: int = 60) -> tuple[np.ndarray, float]:
    """Lloyd's algorithm with k-means++ starts; returns (labels, inertia) of the best of `n_init` restarts."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    x2 = (X ** 2).sum(1)                         # constant over the whole run
    best: tuple[np.ndarray, float] | None = None
    for _ in range(n_init):
        cent = [X[rng.integers(n)]]
        d2 = _sqdist(X, np.array(cent), x2).min(1)
        for _j in range(1, k):
            tot = d2.sum()
            nxt = rng.choice(n, p=d2 / tot) if tot > 0 else rng.integers(n)
            cent.append(X[nxt])
            d2 = np.minimum(d2, _sqdist(X, X[nxt][None, :], x2)[:, 0])
        C = np.array(cent)
        lab = np.zeros(n, dtype=np.int64)
        for it in range(iters):
            D = _sqdist(X, C, x2)
            new = D.argmin(1)
            if it and (new == lab).all():
                break
            lab = new
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = X[m].mean(0)
        inertia = float(_sqdist(X, C, x2)[np.arange(n), lab].sum())
        if best is None or inertia < best[1]:
            best = (lab.copy(), inertia)
    assert best is not None
    return best


def silhouette(X: np.ndarray, lab: np.ndarray, seed: int = 0, max_n: int = SIL_SAMPLE) -> float:
    """Mean silhouette (euclidean) on a deterministic subsample."""
    n = X.shape[0]
    if n > max_n:
        sel = np.random.default_rng(seed).choice(n, max_n, replace=False)
        X, lab = X[sel], lab[sel]
    ks = np.unique(lab)
    if len(ks) < 2:
        return 0.0
    D = np.sqrt(_sqdist(X, X))
    s = np.zeros(len(lab))
    a = np.zeros(len(lab))
    b = np.full(len(lab), np.inf)
    for k in ks:
        m = lab == k
        cnt = m.sum()
        mean_to = D[:, m].sum(1) / np.where(lab == k, max(cnt - 1, 1), cnt)
        a = np.where(m, mean_to, a)
        b = np.where(~m, np.minimum(b, mean_to), b)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return float(s.mean())


# ---------------------------------------------------------------------------------------------
# archetype model
# ---------------------------------------------------------------------------------------------

def deck_matrix(keys: np.ndarray, deck_cards: pl.DataFrame, n_cards: int, spell: np.ndarray) -> sp.csr_matrix:
    """(len(keys), n_cards) copies of spells per deck; `keys` sorted deck keys."""
    dc = deck_cards.select("deck_key", "card_id", "n")
    k = dc["deck_key"].to_numpy().astype(np.int64)
    c = dc["card_id"].to_numpy().astype(np.int64)
    n = dc["n"].to_numpy().astype(float)
    row = np.searchsorted(keys, k)
    ok = (row < len(keys)) & (keys[np.minimum(row, len(keys) - 1)] == k) & spell[c]
    return sp.csr_matrix((n[ok], (row[ok], c[ok])), shape=(len(keys), n_cards))


def l2_rows(X: sp.csr_matrix) -> sp.csr_matrix:
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    return sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ X


@dataclass
class Archetypes:
    n_cards: int
    idf: np.ndarray                      # (n_cards,)
    centroids: np.ndarray                # (A, n_cards) unit rows in TF-IDF space
    info: list[dict] = field(default_factory=list)       # per archetype
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.info)

    # ---- assignment
    def tfidf(self, X: sp.csr_matrix) -> sp.csr_matrix:
        return l2_rows(X @ sp.diags(self.idf))

    def assign_matrix(self, X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """(archetype index, cosine similarity) of every row of a copies matrix."""
        T = self.tfidf(X)
        S = np.asarray((T @ self.centroids.T))
        i = S.argmax(1)
        return i.astype(np.int32), S[np.arange(S.shape[0]), i]

    def vector(self, deck: dict[int, int], spell: np.ndarray) -> sp.csr_matrix:
        ids = [c for c, n in deck.items() if n > 0 and 0 <= c < self.n_cards and spell[c]]
        cnt = [float(deck[c]) for c in ids]
        return sp.csr_matrix((cnt, ([0] * len(ids), ids)), shape=(1, self.n_cards))

    def assign(self, deck: dict[int, int], spell: np.ndarray) -> tuple[int, float, np.ndarray]:
        """(archetype index, similarity, all similarities) for one deck {card_id: copies}."""
        T = self.tfidf(self.vector(deck, spell))
        S = np.asarray(T @ self.centroids.T).ravel()
        i = int(S.argmax()) if len(S) else -1
        return i, float(S[i]) if i >= 0 else 0.0, S

    # ---- persistence
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "archetypes.tmp.npz", idf=self.idf.astype(np.float32), centroids=self.centroids.astype(np.float32))
        import os
        os.replace(d / "archetypes.tmp.npz", d / "archetypes.npz")
        (d / "archetypes.json").write_text(json.dumps(dict(info=self.info, meta=self.meta), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "Archetypes | None":
        try:
            z = np.load(d / "archetypes.npz", allow_pickle=False)
            j = json.loads((d / "archetypes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cls(n_cards=int(z["idf"].shape[0]), idf=z["idf"].astype(float), centroids=z["centroids"].astype(float),
                   info=j["info"], meta=j.get("meta", {}))


def group_label(g: str) -> str:
    return g if g != "other" else "Other"


def build(keys: np.ndarray, X: sp.csr_matrix, groups: np.ndarray, names: list[str], seed: int = 0) -> Archetypes:
    """Cluster the decks (rows of the copies matrix `X`, colour group per row) into archetypes."""
    n_decks, n_cards = X.shape
    df = np.asarray((X > 0).sum(0)).ravel()
    idf = np.log((n_decks + 1.0) / (df + 1.0)) + 1.0
    T = l2_rows(X @ sp.diags(idf)).tocsr()
    mu = np.asarray(T.mean(0)).ravel()
    gram = np.asarray((T.T @ T).todense()) / n_decks - np.outer(mu, mu)
    w, V = np.linalg.eigh(gram)
    V = V[:, np.argsort(-w)[:RANK]]
    E = np.asarray(T @ V) - mu @ V                                # (n_decks, RANK)
    rng_seed = seed
    labels = np.full(n_decks, -1, dtype=np.int64)
    arch_groups: list[str] = []
    sil_report: dict[str, dict] = {}
    group_names = sorted(set(groups.tolist()))
    group_idx = {g: np.nonzero(groups == g)[0] for g in group_names}

    def cluster_group(g):
        """k-means over k = 2..K_MAX inside one colour group (independent of every other group: own seeds, own rows)."""
        idx = group_idx[g]
        Eg = E[idx]
        sils: dict[int, float] = {}
        best_k, best_lab, best_s = 1, np.zeros(len(idx), dtype=np.int64), -1.0
        for k in range(2, K_MAX + 1):
            lab, _ = kmeans(Eg, k, seed=rng_seed + k)
            share = np.bincount(lab, minlength=k) / len(lab)
            s = silhouette(Eg, lab, seed=rng_seed)
            sils[k] = s
            if share.min() >= MIN_SHARE and s > best_s:
                best_k, best_lab, best_s = k, lab, s
        if best_s < SIL_MIN:
            best_k, best_lab = 1, np.zeros(len(idx), dtype=np.int64)
        return best_k, best_lab, sils

    # (sequential: clustering the groups in threads was faster but crashed the process now and then - OpenBLAS is not safe to call
    #  from several Python threads at once on this build)
    done = {g: cluster_group(g) for g in group_names if len(group_idx[g]) >= MIN_GROUP_DECKS}
    for g in group_names:
        idx = group_idx[g]
        if len(idx) < MIN_GROUP_DECKS:
            labels[idx] = len(arch_groups)
            arch_groups.append(g)
            sil_report[g] = dict(n=int(len(idx)), k=1, silhouette={})
            continue
        best_k, best_lab, sils = done[g]
        base = len(arch_groups)
        labels[idx] = base + best_lab
        arch_groups += [g] * best_k
        sil_report[g] = dict(n=int(len(idx)), k=best_k, silhouette={str(k): float(v) for k, v in sils.items()})
    A = len(arch_groups)
    # centroids in TF-IDF space from the cluster members; then reassign every deck to its nearest centroid (the rule used
    # for new decks) and recompute once so that members and centroids agree
    cent = np.zeros((A, n_cards))
    for it in range(2):
        for a in range(A):
            m = np.nonzero(labels == a)[0]
            if len(m):
                v = np.asarray(T[m].mean(0)).ravel()
                cent[a] = v / max(np.linalg.norm(v), 1e-12)
        S = np.asarray(T @ cent.T)
        labels = S.argmax(1)
    sim = S[np.arange(n_decks), labels]
    keep = np.bincount(labels, minlength=A) > 0
    remap = -np.ones(A, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    labels = remap[labels]
    cent = cent[keep]
    arch_groups = [g for g, kp in zip(arch_groups, keep) if kp]
    A = len(arch_groups)
    # signature cards and labels
    B = (X > 0).tocsr()
    info = []
    corpus_rate = np.asarray(B.mean(0)).ravel()
    group_members = {g: np.nonzero([x == g for x in arch_groups])[0] for g in set(arch_groups)}
    for a in range(A):
        m = labels == a
        rate = np.asarray(B[m].mean(0)).ravel()
        g = arch_groups[a]
        if len(group_members[g]) > 1:
            gm = np.isin(labels, group_members[g])
            ref = np.asarray(B[gm].mean(0)).ravel()
        else:
            ref = corpus_rate
        score = rate * np.log2((rate + 0.01) / (ref + 0.01))
        order = [int(c) for c in np.argsort(-score) if rate[c] >= 0.2 and score[c] > 0][:8]
        sig = [dict(card_id=c, name=names[c], rate=float(rate[c]), ref=float(ref[c])) for c in order]
        top = [s["name"].split(" // ")[0] for s in sig[:2]]
        label = group_label(g) + (": " + " / ".join(top) if top else "")
        info.append(dict(id=a, group=g, label=label, n_decks=int(m.sum()), mean_similarity=float(sim[m].mean()),
                         signature=sig, sizes_in_group=int(len(group_members[g]))))
    meta = dict(rank=RANK, k_max=K_MAX, sil_min=SIL_MIN, min_share=MIN_SHARE, groups=sil_report, n_decks=int(n_decks))
    arch = Archetypes(n_cards=n_cards, idf=idf, centroids=cent, info=info, meta=meta)
    arch._labels = labels                      # type: ignore[attr-defined]  (corpus assignment, not persisted)
    arch._sim = sim                            # type: ignore[attr-defined]
    return arch


# ---------------------------------------------------------------------------------------------
# observed win rates
# ---------------------------------------------------------------------------------------------

def attach_win_rates(arch: Archetypes, games: pl.DataFrame, keys: np.ndarray, deck_arch: np.ndarray, seed: int = 0,
                     n_boot: int = N_BOOT) -> None:
    """Per archetype and format: games, wins, raw win rate, shrunk win rate and a cluster-bootstrap 95% CI (events)."""
    rowi = np.searchsorted(keys, games["deck_key"].to_numpy().astype(np.int64))
    rowi = np.minimum(rowi, len(keys) - 1)
    g = games.with_columns(pl.Series("arch", deck_arch[rowi]), pl.col("won").cast(pl.Float64).alias("w"))
    rng = np.random.default_rng(seed)
    base = {f: float(g.filter(pl.col("fmt") == f)["w"].mean()) for f in ("draft", "bo1", "bo3")
            if g.filter(pl.col("fmt") == f).height}
    for a in range(arch.n):
        arch.info[a]["rates"] = {}
    ev = g.group_by("fmt", "arch", "draft_id").agg(pl.col("w").sum().alias("wins"), pl.len().alias("n"))
    # (format, archetype) in a fixed order: polars returns the partitions in hash order, which differs between processes, and the
    # bootstrap draws one after another from `rng` - so the intervals changed from run to run without it
    for (f, a), sub in sorted(ev.partition_by("fmt", "arch", as_dict=True).items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        a = int(a)
        if not (0 <= a < arch.n):
            continue
        wins, n = sub["wins"].to_numpy().astype(float), sub["n"].to_numpy().astype(float)
        N, W = n.sum(), wins.sum()
        p0 = base[f]
        shrunk = (W + SHRINK_GAMES * p0) / (N + SHRINK_GAMES)
        G = len(n)
        if G >= 5:
            mult = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
            Wb, Nb = mult @ wins, mult @ n
            sh = (Wb + SHRINK_GAMES * p0) / (Nb + SHRINK_GAMES)
            lo, hi = np.percentile(sh, [2.5, 97.5])
            rlo, rhi = np.percentile(Wb / np.maximum(Nb, 1), [2.5, 97.5])
        else:
            lo, hi, rlo, rhi = 0.0, 1.0, 0.0, 1.0
        arch.info[a]["rates"][f] = dict(games=int(N), wins=int(W), events=int(G), wr=float(W / N), shrunk=float(shrunk),
                                        lo=float(lo), hi=float(hi), raw_lo=float(rlo), raw_hi=float(rhi), base=float(p0))
    arch.meta["base_rates"] = base
    arch.meta["shrink_games"] = SHRINK_GAMES


def archetype_report(arch: Archetypes) -> dict:
    rows = []
    for a in arch.info:
        rows.append(dict(id=a["id"], label=a["label"], group=a["group"], n_decks=a["n_decks"], rates=a.get("rates", {}),
                         signature=[dict(name=s["name"], rate=s["rate"], ref=s["ref"]) for s in a["signature"][:5]]))
    return dict(n=arch.n, archetypes=rows, groups=arch.meta.get("groups", {}), base_rates=arch.meta.get("base_rates", {}),
                shrink_games=arch.meta.get("shrink_games"), params=dict(rank=RANK, k_max=K_MAX, sil_min=SIL_MIN, min_share=MIN_SHARE))


def columns_for_conditional(deck_arch: np.ndarray, V: sp.csr_matrix, weights: np.ndarray, nb_ids: np.ndarray, n_arch: int,
                            min_games: float = 1500.0, max_cols: int = 2500) -> np.ndarray:
    """(card id, archetype) pairs for the card x archetype columns: cards with >= `min_games` games in decks of the archetype."""
    rows = []
    B = (V > 0).tocsc()
    for a in range(n_arch):
        m = (deck_arch == a) * weights
        g = np.asarray(B.T @ m).ravel()
        for j in np.nonzero(g >= min_games)[0]:
            rows.append((int(nb_ids[j]), a, float(g[j])))
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))
    rows = rows[:max_cols]
    rows.sort(key=lambda r: (r[1], r[0]))
    return np.array([(r[0], r[1]) for r in rows], dtype=np.int64).reshape(-1, 2)
