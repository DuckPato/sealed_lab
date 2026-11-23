"""Where does the split-half disagreement between the two half-models come from?  (diagnostic only: no change to the model)

    python scripts/split_half_decompose.py terms  SH OUT [--sets ECL ...]        (a)+(b): term / card-group shares of the disagreement
    python scripts/split_half_decompose.py stage  SH SRC OUT [--sets MSH SOS FRA HOB] (c): refit the chain per half, draft vs Sealed
    python scripts/split_half_decompose.py report SH OUT                         decompose.json / decompose.md
    python scripts/split_half_decompose.py structure SH DECOMP OUT [--sets ...] [--steps pools refit report]   which structure / mana features carry it
    python scripts/split_half_decompose.py time-one SET HALF SH                  time one rebuild + draft fit (nothing is saved)

SH = the split_half_search OUT folder (homes/SET_A|B, splits/SET.json, search_rows.jsonl); OUT = scratch folder for the outputs; SRC is
not needed any more (the half homes hold a full corpus copy). SEALED_LAB_HOME is always set to a half home (scratch), never the app home.

Definitions (draw 0, one pool = one held-out pool x format).  D_A / D_B = the d0_fast top decks of model A / B.  L_X(D) = logit of model X.
    Delta(D) = L_A(D) - L_B(D)         T = Delta(D_A) - Delta(D_B) = [L_A(D_A) - L_A(D_B)] + [L_B(D_B) - L_B(D_A)]
T is how much more the two models together prefer their own deck: the "self-preference" of split_half_search is T/2 in win-rate points (mean of
the two models' preferences).  L = ref + cards + struct + mana + extra + mull (SetModel._parts / logit); the reference logit is deck independent
and cancels in T exactly.  Every term is split the same way, so shares of T add to 1.
Cards term, per card c, with beta_X the model's beta and w_X(D) the castability-weighted copies (zero when c is not in D):
    contribution_c = beta_A (w_A(D_A) - w_A(D_B)) - beta_B (w_B(D_A) - w_B(D_B))
                   = (beta_A - beta_B) * (dwA + dwB)/2   [beta disagreement times the deck difference]
                   + (beta_A + beta_B)/2 * (dwA - dwB)   [castability-weight disagreement], dwX = w_X(D_A) - w_X(D_B)
Win-rate points: a logit amount x is converted at s = mean of p(1-p) over the four levels L_A(D_A), L_A(D_B), L_B(D_A), L_B(D_B) of the pool and
shown as 100 * s * x / 2 (the /2 puts T on the self-preference scale).  Shares are ratios of sums over pools, 95% intervals are cluster
bootstraps over pools (split_half_search.cluster_boot).

Stage (c): the chain is refitted per half WITHOUT cross-validation, at the sigma / stacking weight the artifact chose (meta.json), on exactly the
half's events with the pipeline's own design code (castability deck space, build_stage, fit_stage, fit_lambda, stack_coef), and the final
betas are checked against the artifact.  beta_bo1 = lam * D + adj1 (adj1 = w1 (M1 - lam D), 0 when the stacking weight is 0); beta_bo3 =
beta_bo1 + adj3 (bo3's prior is the bo1 estimate).  So draft part = lam * D, Sealed part = beta_final - lam * D (bo3: bo1 adjustment + bo3
adjustment).  The draft part is split once more into the common-scale draft disagreement and the lambda (scale) disagreement.

`fit_draft_stage(set_code, home, event_ids, ...)` is the reusable entry point for a draft-stage-only fit on any set of events.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
STAGE_SETS = ["MSH", "SOS", "FRA", "HOB"]
VARIANT = "d0_fast"
TERMS = ("ref", "cards", "struct", "mana", "extra", "mull")
RARITY_GROUPS = ("C", "U", "R", "M", "land", "basic", "other")
TERT_GROUPS = ("T1", "T2", "T3", "basic")
PRES_GROUPS = ("both", "only_A", "only_B")
FMTS = ("bo1", "bo3")
N_BOOT = 2000


# =====================================================================================================================
# pure arithmetic (tested without any model)
# =====================================================================================================================

def card_T_contrib(bA, bB, wA_a, wA_b, wB_a, wB_b):
    """Per-card contribution to the cards term of T and its split into (beta disagreement, castability-weight disagreement).

    bA, bB: betas of the two models; wX_a / wX_b: model X's (castability-weighted) copies in D_A / D_B (0 when absent).
    total = bA (wA_a - wA_b) - bB (wB_a - wB_b) = beta_part + w_part, exactly."""
    bA, bB = np.asarray(bA, float), np.asarray(bB, float)
    dwA = np.asarray(wA_a, float) - np.asarray(wA_b, float)
    dwB = np.asarray(wB_a, float) - np.asarray(wB_b, float)
    total = bA * dwA - bB * dwB
    beta_part = (bA - bB) * (dwA + dwB) / 2.0
    w_part = (bA + bB) / 2.0 * (dwA - dwB)
    return total, beta_part, w_part


def term_T(a_da: float, a_db: float, b_da: float, b_db: float) -> float:
    """T of one additive logit term: [term_A(D_A) - term_A(D_B)] - [term_B(D_A) - term_B(D_B)] = Delta(D_A) - Delta(D_B)."""
    return float((a_da - a_db) - (b_da - b_db))


def group_sums(values, labels, groups) -> dict:
    """Sum of `values` over the entries of each group of `groups` (labels: one per entry). Every label must be a group (no loss)."""
    values = np.asarray(values, float)
    labels = np.asarray(labels, dtype=object)
    out = {g: float(values[labels == g].sum()) for g in groups}
    missing = set(labels.tolist()) - set(groups)
    if missing:
        raise ValueError(f"labels without a group: {sorted(map(str, missing))}")
    return out


def slope(levels) -> float:
    """Mean of p(1-p) at the given logit levels (the logit -> win-rate slope used for the pp conversion)."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(levels, float)))
    return float(np.mean(p * (1.0 - p)))


def exposure_groups(exposure, eligible, labels=("T1", "T2", "T3"), other="basic") -> np.ndarray:
    """Tertile label (low -> high) of each eligible entry by its exposure (ties by rank order, so groups hold equal numbers +- 1); the
    others get `other`."""
    exposure = np.asarray(exposure, float)
    eligible = np.asarray(eligible, bool)
    out = np.full(len(exposure), other, dtype=object)
    idx = np.nonzero(eligible)[0]
    if len(idx):
        order = idx[np.lexsort((idx, exposure[idx]))]
        for k, chunk in enumerate(np.array_split(order, len(labels))):
            out[chunk] = labels[k]
    return out


def presence_classes(n_a, n_b) -> np.ndarray:
    """'both' (card in both decks), 'only_A', 'only_B' (cards in neither deck get 'both': their contribution is 0 anyway)."""
    a, b = np.asarray(n_a) > 0, np.asarray(n_b) > 0
    out = np.full(len(a), "both", dtype=object)
    out[a & ~b] = "only_A"
    out[b & ~a] = "only_B"
    return out


def chain_parts(D, lam1, beta_bo1, beta_bo3, bo1_is_prior: bool = True, lam3=None) -> dict:
    """Split of the final betas into the chain's pieces (arrays over card ids).
    bo1: draft = lam1 * D, bo1_adj = beta_bo1 - lam1 * D.  bo3: draft = lam1 * D, bo1_adj = beta_bo1 - lam1 * D, bo3_adj = beta_bo3 - beta_bo1
    (bo3's prior is the bo1 estimate); when bo1 is no prior (too few games) bo3's prior is lam3 * D."""
    D = np.asarray(D, float)
    out: dict = {}
    if beta_bo1 is not None and lam1 is not None:
        out["bo1"] = dict(draft=lam1 * D, bo1_adj=np.asarray(beta_bo1, float) - lam1 * D, bo3_adj=np.zeros_like(D))
    if beta_bo3 is not None:
        if beta_bo1 is not None and bo1_is_prior and lam1 is not None:
            out["bo3"] = dict(draft=lam1 * D, bo1_adj=np.asarray(beta_bo1, float) - lam1 * D,
                              bo3_adj=np.asarray(beta_bo3, float) - np.asarray(beta_bo1, float))
        else:
            l3 = 1.0 if lam3 is None else lam3
            out["bo3"] = dict(draft=l3 * D, bo1_adj=np.zeros_like(D), bo3_adj=np.asarray(beta_bo3, float) - l3 * D)
    return out


def lambda_split(D_a, D_b, lam_a, lam_b):
    """lam_A D_A and lam_B D_B as (common-scale draft part, scale part): with lam_m = (lam_A + lam_B)/2,
    beta_X = lam_m D_X + (lam_X - lam_m) D_X.  Returns ((bA_common, bB_common), (bA_scale, bB_scale)); each pair sums to lam_X D_X."""
    lm = 0.5 * (lam_a + lam_b)
    D_a, D_b = np.asarray(D_a, float), np.asarray(D_b, float)
    return (lm * D_a, lm * D_b), ((lam_a - lm) * D_a, (lam_b - lm) * D_b)


def weighted_split_moments(v, parts_diff: dict) -> dict:
    """Weighted second moment of d = sum_p d_p with weights v: E[d^2] = sum_p E[d_p^2] + 2 sum_{p<q} E[d_p d_q].
    Returns dict(rms=..., rms_<p>=..., share_<p> = E[d_p^2]/E[d^2], share_cov = 2 sum cross / E[d^2], sum_w=...)."""
    v = np.asarray(v, float)
    names = list(parts_diff)
    tot = sum(np.asarray(parts_diff[n], float) for n in names)
    sw = float(v.sum())
    if sw <= 0:
        return dict(sum_w=0.0)
    m2 = float((v * tot ** 2).sum() / sw)
    out = dict(sum_w=sw, rms=float(np.sqrt(m2)))
    cross = 0.0
    for i, n in enumerate(names):
        di = np.asarray(parts_diff[n], float)
        e = float((v * di ** 2).sum() / sw)
        out[f"rms_{n}"] = float(np.sqrt(e))
        out[f"share_{n}"] = e / m2 if m2 > 0 else float("nan")
        for m in names[i + 1:]:
            cross += 2.0 * float((v * di * np.asarray(parts_diff[m], float)).sum() / sw)
    out["share_cov"] = cross / m2 if m2 > 0 else float("nan")
    return out


def pct(x, nd=0) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


# =====================================================================================================================
# models and card metadata
# =====================================================================================================================

@contextlib.contextmanager
def _home(path):
    old = os.environ.get("SEALED_LAB_HOME")
    os.environ["SEALED_LAB_HOME"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SEALED_LAB_HOME", None)
        else:
            os.environ["SEALED_LAB_HOME"] = old


def home_of(sh: Path, set_code: str, half: str) -> Path:
    return Path(sh) / "homes" / f"{set_code}_{half}"


def load_model(sh: Path, set_code: str, half: str):
    from sealed_lab.model.artifact import SetModel

    with _home(home_of(sh, set_code, half)):
        return SetModel.load(set_code)


def eval_deck(model, deck: dict, fmt: str) -> dict:
    """Draw-0 terms of the logit of `deck` and the per-card effective copies (ids, copies, castability-weighted copies)."""
    ref, cards, struct, mana, _phi, _ids, _cnt = model._parts(deck, fmt, False)
    if fmt in model._extra_fmts:
        pt, at = model._extra_terms(deck, fmt, False)
        extra = float(pt[0] + at[0])
    else:
        extra = 0.0
    mull = float(model.mull_cost(deck, fmt, False)[0])
    ids, cnt, eff = model.deck_weights(deck, fmt)
    beta = model.card_beta(fmt)
    cards_chk = float(beta[ids] @ eff) if len(ids) else 0.0
    out = dict(ref=float(ref[0]), cards=float(cards[0]), struct=float(struct[0]), mana=float(mana[0]), extra=extra, mull=mull,
               ids=np.asarray(ids, np.int64), cnt=np.asarray(cnt, float), eff=np.asarray(eff, float), cards_chk=cards_chk)
    out["L"] = out["ref"] + out["cards"] + out["struct"] + out["mana"] + out["extra"] + out["mull"]
    return out


def card_labels(model, draft_exposure: np.ndarray) -> dict:
    """Per card id: rarity group (C/U/R/M for spells, 'land' = non-basic land, 'basic', 'other') and draft-exposure tertile among the
    non-basic cards (T1 low .. T3 high; 'basic' for basics)."""
    c = model.cards.sort("card_id")
    rar = c["rarity"].to_list()
    is_land, is_basic = c["is_land"].to_numpy(), c["is_basic"].to_numpy()
    lab = []
    for r, il, ib in zip(rar, is_land, is_basic):
        if ib:
            lab.append("basic")
        elif il:
            lab.append("land")
        else:
            lab.append({"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}.get(str(r), "other"))
    ex = np.asarray(draft_exposure, float)
    return dict(rarity=np.array(lab, dtype=object), tertile=exposure_groups(ex, ~is_basic), exposure=ex,
                names=c["name"].to_list())


def draft_exposure_from_corpus(corpus_dir: Path, event_ids, n_cards: int) -> np.ndarray:
    """Draft exposure per card id = sum over the draft games of the given events of the card's copies in the deck (the StageData
    definition: copies x games), computed from the corpus (no fit)."""
    import polars as pl

    g = pl.read_parquet(Path(corpus_dir) / "games.parquet", columns=["fmt", "draft_id", "deck_key"]).filter(pl.col("fmt") == "draft")
    g = g.filter(pl.col("draft_id").cast(pl.Utf8).is_in([str(e) for e in event_ids]))
    per_deck = g.group_by("deck_key").len()
    dc = pl.read_parquet(Path(corpus_dir) / "deck_cards.parquet").join(per_deck, on="deck_key", how="inner")
    agg = dc.group_by("card_id").agg((pl.col("n") * pl.col("len")).sum().alias("e"))
    out = np.zeros(n_cards)
    out[agg["card_id"].to_numpy()] = agg["e"].to_numpy().astype(float)
    return out


def split_events_of(sh: Path, set_code: str) -> dict:
    return json.loads((Path(sh) / "splits" / f"{set_code}.json").read_text(encoding="utf-8"))


# =====================================================================================================================
# (a) + (b): terms and card groups, per pool
# =====================================================================================================================

def pool_record(row: dict, mA, mB, lab: dict) -> dict:
    fmt = row["fmt"]
    DA = {int(k): int(v) for k, v in row["dirs"]["A"][VARIANT]["deck"].items()}
    DB = {int(k): int(v) for k, v in row["dirs"]["B"][VARIANT]["deck"].items()}
    e = {("A", "A"): eval_deck(mA, DA, fmt), ("A", "B"): eval_deck(mA, DB, fmt),
         ("B", "A"): eval_deck(mB, DA, fmt), ("B", "B"): eval_deck(mB, DB, fmt)}      # (model, deck)
    rec = dict(set=row["set"], fmt=fmt, pool=f'{row["set"]}|{fmt}|{row["draft_id"]}')
    # consistency with the stored search rows and with the model's own logit
    rec["chk_stored"] = max(abs(e[("A", "A")]["L"] - row["dirs"]["A"][VARIANT]["ls"]), abs(e[("B", "A")]["L"] - row["dirs"]["A"][VARIANT]["lo"]),
                            abs(e[("B", "B")]["L"] - row["dirs"]["B"][VARIANT]["ls"]), abs(e[("A", "B")]["L"] - row["dirs"]["B"][VARIANT]["lo"]))
    rec["chk_cards"] = max(abs(v["cards"] - v["cards_chk"]) for v in e.values())
    rec["chk_logit"] = abs(e[("A", "A")]["L"] - float(mA.logit(DA, fmt)))
    L = {k: v["L"] for k, v in e.items()}
    rec.update(LA_DA=L[("A", "A")], LA_DB=L[("A", "B")], LB_DA=L[("B", "A")], LB_DB=L[("B", "B")])
    rec["T"] = term_T(L[("A", "A")], L[("A", "B")], L[("B", "A")], L[("B", "B")])
    rec["T_dirA"] = L[("A", "A")] - L[("A", "B")]          # what model A gains by its own deck
    rec["T_dirB"] = L[("B", "B")] - L[("B", "A")]
    rec["terms"] = {t: term_T(e[("A", "A")][t], e[("A", "B")][t], e[("B", "A")][t], e[("B", "B")][t]) for t in TERMS}
    rec["s"] = slope([L[("A", "A")], L[("A", "B")], L[("B", "A")], L[("B", "B")]])
    pa = lambda x: 1.0 / (1.0 + np.exp(-x))
    rec["pref_pp"] = 100.0 * 0.5 * ((pa(L[("A", "A")]) - pa(L[("A", "B")])) + (pa(L[("B", "B")]) - pa(L[("B", "A")])))
    # per-card contributions over the union of the decks' cards
    ids = np.unique(np.concatenate([v["ids"] for v in e.values()]))
    n = len(lab["rarity"])

    def dense(x):
        out = np.zeros(n)
        out[x["ids"]] = x["eff"]
        return out

    def dense_cnt(x):
        out = np.zeros(n)
        out[x["ids"]] = x["cnt"]
        return out

    wA_a, wA_b, wB_a, wB_b = (dense(e[("A", "A")])[ids], dense(e[("A", "B")])[ids], dense(e[("B", "A")])[ids], dense(e[("B", "B")])[ids])
    bA, bB = mA.card_beta(fmt)[ids], mB.card_beta(fmt)[ids]
    cnt_a, cnt_b = dense_cnt(e[("A", "A")])[ids], dense_cnt(e[("A", "B")])[ids]
    tot, bpart, wpart = card_T_contrib(bA, bB, wA_a, wA_b, wB_a, wB_b)
    rec["cards_check"] = float(abs(tot.sum() - rec["terms"]["cards"]))
    pres = presence_classes(cnt_a, cnt_b)
    rec["groups"] = dict(rarity=group_sums(tot, lab["rarity"][ids], RARITY_GROUPS), tertile=group_sums(tot, lab["tertile"][ids], TERT_GROUPS),
                         presence=group_sums(tot, pres, PRES_GROUPS), beta_part=float(bpart.sum()), w_part=float(wpart.sum()),
                         beta_part_both=float(bpart[pres == "both"].sum()), w_part_both=float(wpart[pres == "both"].sum()))
    rec["arrays"] = dict(ids=ids, bA=bA, bB=bB, wA_a=wA_a, wA_b=wA_b, wB_a=wB_a, wB_b=wB_b, cnt_a=cnt_a, cnt_b=cnt_b)
    return rec


def run_terms(sh: Path, out: Path, sets: list[str]) -> None:
    rows_all = [json.loads(x) for x in (sh / "search_rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (out / "terms").mkdir(parents=True, exist_ok=True)
    for sc in sets:
        t0 = time.time()
        rows = [r for r in rows_all if r["set"] == sc]
        mA, mB = load_model(sh, sc, "A"), load_model(sh, sc, "B")
        ev = split_events_of(sh, sc)
        expo = draft_exposure_from_corpus(home_of(sh, sc, "A") / "sets" / sc / "corpus", set(ev["A"]) | set(ev["B"]), len(mA.cards["card_id"]))
        lab = card_labels(mA, expo)
        recs = []
        with _home(home_of(sh, sc, "A")):                                    # (the cast cache lives in the model object)
            for i, r in enumerate(rows):
                recs.append(pool_record(r, mA, mB, lab))
                if (i + 1) % 10 == 0:
                    print(f"  {sc}: {i + 1}/{len(rows)} pools, {time.time() - t0:.0f}s", flush=True)
        with open(out / "terms" / f"{sc}.pkl", "wb") as fh:
            pickle.dump(dict(set=sc, records=recs, labels=lab, n_ev=(len(ev["A"]), len(ev["B"]))), fh)
        print(f"terms {sc}: {len(recs)} pools in {time.time() - t0:.0f}s; max|stored-recomputed logit| {max(r['chk_stored'] for r in recs):.2e}, "
              f"max cards check {max(r['cards_check'] for r in recs):.2e}", flush=True)


# =====================================================================================================================
# (c): rebuild the half's design and refit the chain at the artifact's hyper-parameters
# =====================================================================================================================

class HalfSetup:
    """The pipeline's design objects for one set on a subset of events (castability deck space, the stages)."""

    def __init__(self, set_code: str, home: Path, event_ids, stages=("draft", "bo1", "bo3"), n_jobs: int = 1, allow_plain: bool = False):
        import polars as pl

        from sealed_lab import config
        from sealed_lab.model import castability as cast_mod
        from sealed_lab.model import pipeline, structure
        from sealed_lab.model.design import build_deck_space, build_stage

        self.set_code, self.home = set_code, Path(home)
        self.meta = json.loads((self.home / "sets" / set_code / "model" / "meta.json").read_text(encoding="utf-8"))
        ma = self.meta.get("manasim") or {}
        if ma.get("applied"):
            raise NotImplementedError("the half model applied the v2 mana simulator: not supported by this rebuild")
        cast_on = bool((self.meta.get("castability") or {}).get("applied"))
        if not cast_on and not allow_plain:
            raise NotImplementedError("castability not applied in this half model: not supported by this rebuild")
        keep = set(str(e) for e in event_ids)
        with _home(self.home):
            from sealed_lab.model.manasim import SimRegime

            inp = pipeline.load_inputs(set_code)
            inp.games = inp.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in(keep))
            games = inp.games
            inp.cards = cast_mod.with_cost_groups(inp.cards, set_code)
            sim_regimes = {r: SimRegime.make(r, k) for r, k in ((ma.get("keep") or {}).items())}
            keys, table = pipeline.phi_for_corpus(inp, n_jobs, None, None, config.cache_dir(set_code))
            phi = table[:, :structure.N_FEATURES]
            per_deck = games.group_by("deck_key").len().sort("deck_key")
            wts = np.zeros(len(keys))
            wts[np.searchsorted(keys, per_deck["deck_key"].to_numpy())] = per_deck["len"].to_numpy()
            space = build_deck_space(keys, phi, inp.deck_cards, inp.cards, weights=np.maximum(wts, 1e-9))
            if cast_on:
                cast_tabs = {}
                for r_, m_ in pipeline.regime_masks(games, keys).items():
                    cast_tabs[r_] = pipeline.cast_for_corpus(inp, sim_regimes.get(r_, r_), n_jobs, None, None, config.cache_dir(set_code), need=m_)
                self.fspace = pipeline.cast_space(inp, games, keys, {r_: table for r_ in cast_tabs}, cast_tabs)
            else:                                            # plain chain (allow_plain): the pipeline fits on the plain deck space
                self.fspace = space
        self.space, self.inp = space, inp
        self.n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
        self.stages = {n: build_stage(n, games.filter(pl.col("fmt") == n), self.fspace) for n in stages if self.n_games.get(n, 0) > 0}
        self.n_card, self.n_feat = space.n_card, self.fspace.n_feat
        self.nb_ids = space.nb_ids
        self.n_cards_total = space.n_cards_total

    def to_full(self, coef_cards: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_cards_total)
        out[self.nb_ids] = coef_cards
        return out


def fit_draft_stage(set_code: str, home, event_ids, sigma: float | None = None, with_se: bool = True, setup: HalfSetup | None = None) -> dict:
    """Draft-stage fit on exactly the given events (draft games of those events), as the pipeline's final fit does it (castability deck
    space, zero prior mean, no attributes). `sigma`: the card prior SD; None = the one the home's artifact chose for the draft stage
    (meta.json prior_strength) - fixed, so fits on different event sets are comparable (no cross-validation here).
    Returns dict(beta (n_cards_total, 0 for basics), se (same shape, NaN without SEs), exposure (per card id: copies x games),
    sigma, n_games, n_events, converged, setup)."""
    from sealed_lab.model import cardvalue

    st_setup = setup or HalfSetup(set_code, Path(home), event_ids, stages=("draft",))
    if "draft" not in st_setup.stages:
        raise ValueError("no draft games in these events")
    st = st_setup.stages["draft"]
    if sigma is None:
        sigma = float(st_setup.meta["prior_strength"]["stages"]["draft"]["sigma"])
    spec = cardvalue.make_spec(st, st_setup.n_card, st_setup.n_feat, "draft")
    prior = np.zeros(st_setup.n_card + st_setup.n_feat)
    coef, se, res = cardvalue.fit_stage(st, prior, spec, sigma, with_se=with_se)
    k = st.prob.k
    beta = st_setup.to_full(coef[k:k + st_setup.n_card])
    se_full = np.full(st_setup.n_cards_total, np.nan)
    if se is not None:
        se_full[st_setup.nb_ids] = np.asarray(se[1])[k:k + st_setup.n_card]
    expo = st_setup.to_full(st.exposure)
    return dict(beta=beta, se=se_full, exposure=expo, sigma=float(sigma), n_games=int(st.n_games), n_events=int(st.n_groups),
                converged=bool(res.converged), setup=st_setup)


def refit_chain_fixed(setup: HalfSetup) -> dict:
    """run_chain without cross-validation: each stage at the sigma / stacking weight the artifact chose. Returns per stage
    dict(coef, k, lam, w, sigma, prior_deck)."""
    from sealed_lab.model import cardvalue, glm, validate

    ps = setup.meta["prior_strength"]["stages"]
    formats = list(setup.meta["formats"])
    prior_names = {n for n in ("draft", "bo1") if setup.n_games.get(n, 0) >= validate.MIN_PRIOR_GAMES}
    prior_ok = {n for n in setup.stages if n in prior_names}
    p_deck = setup.n_card + setup.n_feat
    fits: dict = {}
    prev = None
    for name in [n for n in ("draft", "bo1", "bo3") if n in setup.stages]:
        st = setup.stages[name]
        lam = lam_raw = None
        if prev is None:
            prior = np.zeros(p_deck)
        elif prev["name"] == "draft":
            lam_raw, _ = cardvalue.fit_lambda(st, prev["deck"])
            lam = float(np.clip(lam_raw, *cardvalue.LAMBDA_CLIP))
            prior = lam * prev["deck"]
        else:
            prior = prev["deck"].copy()
        sigma, w = float(ps[name]["sigma"]), float(ps[name]["w"])
        spec = cardvalue.make_spec(st, setup.n_card, setup.n_feat, name)
        coef, _, res = cardvalue.fit_stage(st, prior, spec, sigma, with_se=False)
        if w < 1.0:
            coef = glm.stack_coef(st.prob, coef, prior, w, spec.sig_cov, st.w)
        fits[name] = dict(coef=coef, k=st.prob.k, lam=lam, lam_raw=lam_raw, w=w, sigma=sigma, prior_deck=prior, converged=bool(res.converged),
                          exposure=st.exposure)
        if name in prior_ok:
            prev = dict(name=name, deck=coef[st.prob.k:])
    fits["_formats"] = formats
    fits["_prior_ok"] = sorted(prior_ok)
    return fits


def stage_for_half(sh: Path, sc: str, half: str, out: Path) -> dict:
    """Refit one half's chain and check it against the artifact; saves OUT/stage/SET_H.npz (+ .json)."""
    t0 = time.time()
    ev = split_events_of(sh, sc)
    setup = HalfSetup(sc, home_of(sh, sc, half), ev[half])
    t_setup = time.time() - t0
    t1 = time.time()
    fits = refit_chain_fixed(setup)
    t_fit = time.time() - t1
    model = load_model(sh, sc, half)
    info: dict = dict(set=sc, half=half, seconds_setup=t_setup, seconds_fit=t_fit, formats=fits["_formats"], prior_ok=fits["_prior_ok"], recon={})
    arrays: dict = {}
    k_d = fits["draft"]["k"]
    D = setup.to_full(fits["draft"]["coef"][k_d:k_d + setup.n_card])
    arrays["D"] = D
    arrays["exposure_draft_half"] = setup.to_full(fits["draft"]["exposure"])
    dfit = fit_draft_stage(sc, home_of(sh, sc, half), ev[half], with_se=True, setup=setup)      # the reusable entry point (also a check)
    info["draft_fit_vs_chain_max_abs"] = float(np.abs(dfit["beta"] - D).max())
    arrays["se_draft"] = np.nan_to_num(dfit["se"], nan=0.0)
    betas = {}
    for f in fits["_formats"]:
        ft = fits[f]
        b = setup.to_full(ft["coef"][ft["k"]:ft["k"] + setup.n_card])
        art = model.card_beta(f)
        nb = setup.nb_ids
        err = b[nb] - art[nb]
        th_rec = ft["coef"][ft["k"] + setup.n_card:ft["k"] + setup.n_card + setup.n_feat]
        th_art = model._draws[f]["theta"][0]
        meta_st = setup.meta["stages"][f]
        info["recon"][f] = dict(max_abs=float(np.abs(err).max()), rms=float(np.sqrt(np.mean(err ** 2))), art_rms=float(np.sqrt(np.mean(art[nb] ** 2))),
                                theta_max_abs=float(np.abs(th_rec - th_art).max()), lam_rec=ft["lam"], lam_art=meta_st.get("lam"), w=ft["w"],
                                sigma=ft["sigma"], converged=ft["converged"])
        betas[f] = (b, art)
        arrays[f"beta_rec_{f}"] = b
        arrays[f"beta_art_{f}"] = art
    info["lam_draft_to_bo1"] = fits["bo1"]["lam"] if "bo1" in fits else None
    info["lam_draft_to_bo3"] = fits["bo3"]["lam"] if "bo3" in fits else None
    # the pieces (reconstructed); `resid` = artifact beta - sum of the reconstructed pieces
    lam1 = fits["bo1"]["lam"] if "bo1" in fits else None
    bo1_prior = "bo1" in fits["_prior_ok"]
    p = chain_parts(D, lam1, betas["bo1"][0] if "bo1" in betas else None, betas["bo3"][0] if "bo3" in betas else None, bo1_prior,
                    fits["bo3"]["lam"] if "bo3" in fits else None)
    for f, parts in p.items():
        for nm, v in parts.items():
            arrays[f"part_{f}_{nm}"] = v
        arrays[f"part_{f}_resid"] = betas[f][1] - sum(parts.values())
    arrays["lam1"] = np.array([np.nan if lam1 is None else lam1])
    (out / "stage").mkdir(parents=True, exist_ok=True)
    np.savez(out / "stage" / f"{sc}_{half}.npz", **arrays)
    (out / "stage" / f"{sc}_{half}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"stage {sc} {half}: setup {t_setup:.0f}s, fit {t_fit:.0f}s; recon "
          + ", ".join(f"{f}: max {v['max_abs']:.2e} rms {v['rms']:.2e}" for f, v in info["recon"].items()), flush=True)
    return info


# =====================================================================================================================
# report
# =====================================================================================================================

def _boot(num, den, clusters):
    from split_half_search import cluster_boot

    return cluster_boot(num, den, clusters, N_BOOT, 0)


def _cell(recs, num_fn, den_fn=lambda r: r["T"]):
    if not recs:
        return None
    pt, lo, hi = _boot([num_fn(r) for r in recs], [den_fn(r) for r in recs], [r["pool"] for r in recs])
    return dict(share=pt, lo=lo, hi=hi)


def _fmt_share(c):
    return "-" if c is None else f"{pct(c['share'], 1)} [{pct(c['lo'], 1)}, {pct(c['hi'], 1)}]"


def load_terms(out: Path, sets: list[str]) -> dict:
    res = {}
    for sc in sets:
        p = out / "terms" / f"{sc}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                res[sc] = pickle.load(fh)
    return res


def selections(recs: list[dict]) -> dict:
    sel = {"ALL": recs, "bo1": [r for r in recs if r["fmt"] == "bo1"], "bo3": [r for r in recs if r["fmt"] == "bo3"]}
    return sel


def summarise_terms(recs: list[dict]) -> dict:
    """Shares of T for every term and card group in one selection of pool records."""
    if not recs:
        return {}
    T = np.array([r["T"] for r in recs])
    s = np.array([r["s"] for r in recs])
    out: dict = dict(n_pools=len(recs), mean_T_logit=float(T.mean()), mean_T_half_pp_lin=float(np.mean(100 * s * T / 2)),
                     mean_pref_pp=float(np.mean([r["pref_pp"] for r in recs])), mean_slope=float(s.mean()))
    pref = _boot([r["pref_pp"] for r in recs], np.ones(len(recs)), [r["pool"] for r in recs])
    out["pref_pp_ci"] = [pref[1], pref[2]]
    out["terms"] = {t: _cell(recs, lambda r, t=t: r["terms"][t]) for t in TERMS}
    out["terms"]["structure (struct+mana)"] = _cell(recs, lambda r: r["terms"]["struct"] + r["terms"]["mana"])
    out["terms"]["extra+mull"] = _cell(recs, lambda r: r["terms"]["extra"] + r["terms"]["mull"])
    out["terms_pp"] = {t: float(np.mean([50 * r["s"] * r["terms"][t] for r in recs])) for t in TERMS}
    out["rarity"] = {g: _cell(recs, lambda r, g=g: r["groups"]["rarity"][g]) for g in RARITY_GROUPS}
    out["tertile"] = {g: _cell(recs, lambda r, g=g: r["groups"]["tertile"][g]) for g in TERT_GROUPS}
    out["presence"] = {g: _cell(recs, lambda r, g=g: r["groups"]["presence"][g]) for g in PRES_GROUPS}
    out["cards_beta_part"] = _cell(recs, lambda r: r["groups"]["beta_part"])
    out["cards_w_part"] = _cell(recs, lambda r: r["groups"]["w_part"])
    out["cards_beta_part_both"] = _cell(recs, lambda r: r["groups"]["beta_part_both"])
    out["cards_w_part_both"] = _cell(recs, lambda r: r["groups"]["w_part_both"])
    out["cards_pp"] = float(np.mean([50 * r["s"] * r["terms"]["cards"] for r in recs]))
    return out


def stage_pool_parts(rec: dict, partsA: dict, partsB: dict, lab: dict, lamsplit: tuple | None = None) -> dict:
    """T cards-term contribution of each chain piece for one pool record (from its stored per-card weights): dict(piece -> total),
    plus by-group sums for 'draft' and 'sealed'."""
    a = rec["arrays"]
    ids = a["ids"]
    res: dict = {}
    per_card: dict = {}
    for nm in partsA:
        tot, _, _ = card_T_contrib(partsA[nm][ids], partsB[nm][ids], a["wA_a"], a["wA_b"], a["wB_a"], a["wB_b"])
        per_card[nm] = tot
        res[nm] = float(tot.sum())
    if lamsplit is not None:
        (cA, cB), (sA, sB) = lamsplit
        tot_c, _, _ = card_T_contrib(cA[ids], cB[ids], a["wA_a"], a["wA_b"], a["wB_a"], a["wB_b"])
        tot_s, _, _ = card_T_contrib(sA[ids], sB[ids], a["wA_a"], a["wA_b"], a["wB_a"], a["wB_b"])
        res["draft_common"], res["draft_scale"] = float(tot_c.sum()), float(tot_s.sum())
    sealed = sum(per_card[n] for n in per_card if n not in ("draft", "resid"))
    per_card["sealed"] = sealed
    res["sealed"] = float(sealed.sum())
    res["by_rarity"] = {pc: group_sums(per_card[pc], lab["rarity"][ids], RARITY_GROUPS) for pc in ("draft", "sealed")}
    res["by_tertile"] = {pc: group_sums(per_card[pc], lab["tertile"][ids], TERT_GROUPS) for pc in ("draft", "sealed")}
    return res


def run_report(sh: Path, out: Path, sets: list[str], stage_sets: list[str]) -> str:
    terms = load_terms(out, sets)
    sets = [s for s in sets if s in terms]
    recs_all = [r for s in sets for r in terms[s]["records"]]
    res: dict = dict(sets=sets, n_pools=len(recs_all), variant=VARIANT, n_boot=N_BOOT)
    res["checks"] = dict(max_stored_logit_diff=max(r["chk_stored"] for r in recs_all), max_cards_vs_parts=max(r["chk_cards"] for r in recs_all),
                         max_logit_vs_sum_of_terms=max(abs(r["T"] - sum(r["terms"].values())) for r in recs_all),
                         max_cards_groups_sum=max(r["cards_check"] for r in recs_all))
    res["terms"] = {}
    for name, rs in selections(recs_all).items():
        res["terms"][name] = summarise_terms(rs)
    res["terms_per_set"] = {s: {f: summarise_terms([r for r in terms[s]["records"] if f == "ALL" or r["fmt"] == f]) for f in ("ALL", "bo1", "bo3")}
                            for s in sets}
    # reproduce the split_half figure from the stored rows (default search) and the same for d0_fast
    rows_all = [json.loads(x) for x in (Path(sh) / "search_rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    res["repro"] = {}
    ex = lambda x: 1.0 / (1.0 + np.exp(-x))
    for var in ("default", "d0_fast"):
        pr = []
        for r in rows_all:
            a, b = r["dirs"]["A"][var], r["dirs"]["B"][var]
            pr.append(dict(pool=f'{r["set"]}|{r["fmt"]}|{r["draft_id"]}', fmt=r["fmt"], pref=100 * 0.5 * ((ex(a["ls"]) - ex(b["lo"])) + (ex(b["ls"]) - ex(a["lo"])))))
        res["repro"][var] = {}
        for k, sel in (("ALL", pr), ("bo1", [x for x in pr if x["fmt"] == "bo1"]), ("bo3", [x for x in pr if x["fmt"] == "bo3"])):
            c = _boot([x["pref"] for x in sel], np.ones(len(sel)), [x["pool"] for x in sel])
            res["repro"][var][k] = dict(mean=c[0], lo=c[1], hi=c[2], n=len(sel))
    sj = Path(sh) / "split_half.json"
    if sj.exists():
        d = json.loads(sj.read_text(encoding="utf-8"))["summary"]["default_vs_default"]
        res["split_half_reference_default_search"] = {k: d[k].get("mean_self_preference_pp") for k in ("overall", "bo1", "bo3")}
    res["stage"] = run_stage_report(out, terms, stage_sets)
    (out / "decompose.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    md = markdown(res)
    (out / "decompose.md").write_text(md, encoding="utf-8")
    return md


def run_stage_report(out: Path, terms: dict, stage_sets: list[str]) -> dict:
    res: dict = {"sets": [], "recon": {}, "per_set": {}}
    pooled: dict = {f: {"recs": [], "var": {}} for f in FMTS}
    for sc in stage_sets:
        pa, pb = out / "stage" / f"{sc}_A.npz", out / "stage" / f"{sc}_B.npz"
        if sc not in terms or not pa.exists() or not pb.exists():
            continue
        res["sets"].append(sc)
        A, B = dict(np.load(pa)), dict(np.load(pb))
        infoA = json.loads((out / "stage" / f"{sc}_A.json").read_text(encoding="utf-8"))
        infoB = json.loads((out / "stage" / f"{sc}_B.json").read_text(encoding="utf-8"))
        res["recon"][sc] = dict(A=infoA["recon"], B=infoB["recon"], seconds={"A": [infoA["seconds_setup"], infoA["seconds_fit"]],
                                                                            "B": [infoB["seconds_setup"], infoB["seconds_fit"]]})
        lab = terms[sc]["labels"]
        lamA, lamB = float(A["lam1"][0]), float(B["lam1"][0])
        ls = lambda_split(A["D"], B["D"], lamA, lamB) if np.isfinite(lamA) and np.isfinite(lamB) else None
        res["per_set"][sc] = {}
        for f in FMTS:
            key = lambda n, H: H.get(f"part_{f}_{n}")
            names = [n for n in ("draft", "bo1_adj", "bo3_adj", "resid") if key(n, A) is not None and key(n, B) is not None]
            if not names:
                continue
            partsA = {n: key(n, A) for n in names}
            partsB = {n: key(n, B) for n in names}
            recs = [r for r in terms[sc]["records"] if r["fmt"] == f]
            rows = []
            for r in recs:
                sp_ = stage_pool_parts(r, partsA, partsB, lab, ls)
                rows.append(dict(pool=r["pool"], T=r["T"], T_cards=r["terms"]["cards"], s=r["s"],
                                 **{"p_" + k: v for k, v in sp_.items() if not isinstance(v, dict)},
                                 by_rarity=sp_["by_rarity"], by_tertile=sp_["by_tertile"]))
            # card-level disagreement weighted by how often the card is in the disagreement (|count diff| summed over pools)
            n = len(lab["rarity"])
            v = np.zeros(n)
            for r in recs:
                a = r["arrays"]
                np.add.at(v, a["ids"], np.abs(a["cnt_a"] - a["cnt_b"]))
            dparts = {nm: partsA[nm] - partsB[nm] for nm in names}
            sealed = {nm: d for nm, d in dparts.items() if nm in ("bo1_adj", "bo3_adj")}
            groups = {"ALL": np.ones(n, bool)}
            groups.update({f"rar:{g}": lab["rarity"] == g for g in RARITY_GROUPS})
            groups.update({f"exp:{g}": lab["tertile"] == g for g in TERT_GROUPS})
            var = {}
            for gname, m in groups.items():
                if not m.any() or v[m].sum() <= 0:
                    continue
                two = {"draft": dparts["draft"][m], "sealed": sum(sealed.values())[m] if sealed else np.zeros(m.sum())}
                var[gname] = weighted_split_moments(v[m], two)
                if f == "bo3" and "bo1_adj" in dparts:
                    var[gname + "|3way"] = weighted_split_moments(v[m], {"draft": dparts["draft"][m], "bo1_adj": dparts["bo1_adj"][m],
                                                                         "bo3_adj": dparts["bo3_adj"][m]})
            lam_m = 0.5 * (lamA + lamB) if np.isfinite(lamA) and np.isfinite(lamB) else 1.0
            nz = lam_m * np.sqrt(A["se_draft"] ** 2 + B["se_draft"] ** 2)          # sampling-noise scale of d_draft = lam_m (D_A - D_B)
            for gname, m in groups.items():
                if gname in var and v[m].sum() > 0:
                    var[gname]["rms_noise"] = float(np.sqrt((v[m] * nz[m] ** 2).sum() / v[m].sum()))
            res["per_set"][sc][f] = dict(rows=rows, var=var, v=v, dparts=dparts, names=names, lam=(lamA, lamB), nz=nz)
            pooled[f]["recs"].append((sc, rows))
            pooled[f]["var"][sc] = (v, dparts, lab, nz)
    # summaries (strip the bulky pieces from what is stored in json)
    summ: dict = {}
    for f in FMTS:
        allrows = [r for _sc, rows in pooled[f]["recs"] for r in rows]
        if not allrows:
            continue
        piece = lambda k: _cell(allrows, lambda r, k=k: r.get("p_" + k, 0.0))
        s_f: dict = dict(n_pools=len(allrows), cards_share_of_T=_cell(allrows, lambda r: r["T_cards"]))
        for k in ("draft", "bo1_adj", "bo3_adj", "sealed", "resid", "draft_common", "draft_scale"):
            if any(("p_" + k) in r for r in allrows):
                s_f["share_" + k] = piece(k)
        for gk, groups in (("by_rarity", RARITY_GROUPS), ("by_tertile", TERT_GROUPS)):
            s_f[gk] = {pc: {g: _cell(allrows, lambda r, pc=pc, g=g, gk=gk: r[gk][pc][g]) for g in groups} for pc in ("draft", "sealed")}
        # variance of beta_A - beta_B, pooled over the sets (weights summed over sets)
        vv: dict = {}
        for gname in ["ALL"] + [f"rar:{g}" for g in RARITY_GROUPS] + [f"exp:{g}" for g in TERT_GROUPS]:
            vs, dd, ss = [], {"draft": [], "sealed": []}, []
            for sc, (v, dparts, lab, nz) in pooled[f]["var"].items():
                m = np.ones(len(v), bool) if gname == "ALL" else ((lab["rarity"] == gname[4:]) if gname.startswith("rar:") else (lab["tertile"] == gname[4:]))
                vs.append(v[m])
                ss.append(nz[m])
                dd["draft"].append(dparts["draft"][m])
                dd["sealed"].append(sum(d for nm, d in dparts.items() if nm in ("bo1_adj", "bo3_adj"))[m] if any(nm in dparts for nm in ("bo1_adj", "bo3_adj")) else np.zeros(m.sum()))
            vcat = np.concatenate(vs)
            if vcat.sum() > 0:
                vv[gname] = weighted_split_moments(vcat, {k: np.concatenate(x) for k, x in dd.items()})
                vv[gname]["rms_noise"] = float(np.sqrt((vcat * np.concatenate(ss) ** 2).sum() / vcat.sum()))
        s_f["var"] = vv
        summ[f] = s_f
    res["pooled"] = summ
    # per-set compact numbers
    res["per_set_compact"] = {}
    for sc in res["sets"]:
        res["per_set_compact"][sc] = {}
        for f, d in res["per_set"][sc].items():
            rows = d["rows"]
            tot = sum(r["T_cards"] for r in rows)
            res["per_set_compact"][sc][f] = dict(
                n_pools=len(rows), cards_T=tot, share_draft=sum(r.get("p_draft", 0.0) for r in rows) / tot if tot else float("nan"),
                share_sealed=sum(r.get("p_sealed", 0.0) for r in rows) / tot if tot else float("nan"),
                rms_dbeta=d["var"].get("ALL", {}).get("rms"), rms_noise=d["var"].get("ALL", {}).get("rms_noise"), share_var_draft=d["var"].get("ALL", {}).get("share_draft"),
                share_var_sealed=d["var"].get("ALL", {}).get("share_sealed"), share_var_cov=d["var"].get("ALL", {}).get("share_cov"),
                lam_A=d["lam"][0], lam_B=d["lam"][1])
    for sc in list(res["per_set"]):                       # bulky arrays out of the json
        for f in list(res["per_set"][sc]):
            res["per_set"][sc][f] = dict(var=res["per_set"][sc][f]["var"])
    return res


def markdown(res: dict) -> str:
    L = ["# Where the split-half disagreement comes from", "",
         f"d0_fast top decks of model A / B on {res['n_pools']} pool x format cells ({len(res['sets'])} sets), draw 0. T = [L_A(D_A) - L_A(D_B)] + [L_B(D_B) - L_B(D_A)] "
         "(logit); T/2 is the self-preference of split_half_search. Shares = sum over pools of the component / sum of T, 95% cluster bootstrap over pools "
         f"({res['n_boot']} resamples). Win-rate points: 100 * s * x / 2 with s = mean p(1-p) at the four deck levels of the pool.", ""]
    ck = res["checks"]
    L += [f"Checks: max |stored - recomputed logit| {ck['max_stored_logit_diff']:.1e}; max |cards term - sum of per-card parts| {ck['max_cards_vs_parts']:.1e}; "
          f"max |T - sum of terms| {ck['max_logit_vs_sum_of_terms']:.1e}; max |cards T - sum of card groups| {ck['max_cards_groups_sum']:.1e}.", ""]
    ref = res.get("split_half_reference_default_search")
    t = res["terms"]
    rp = res["repro"]
    L += ["## Self-preference reproduction (pp)", "",
          "| cell | `default` search, recomputed from stored rows [95% CI] | split_half.json (`default`) | `d0_fast` (used below) [95% CI] | d0_fast from re-evaluated models | mean T (logit), d0_fast | T/2 via slope pp, d0_fast |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in ("ALL", "bo1", "bo3"):
        d = t[k]
        r0, r1 = rp["default"][k], rp["d0_fast"][k]
        L.append(f"| {k} | {r0['mean']:.2f} [{r0['lo']:.2f}, {r0['hi']:.2f}] | {'-' if not ref else format(ref['overall' if k == 'ALL' else k], '.2f')} | "
                 f"{r1['mean']:.2f} [{r1['lo']:.2f}, {r1['hi']:.2f}] | {d['mean_pref_pp']:.2f} | {d['mean_T_logit']:.3f} | {d['mean_T_half_pp_lin']:.2f} |")
    L += [""]
    L += ["## (a) Share of T by logit term", "", "| term | ALL | bo1 | bo3 | mean pp ALL / bo1 / bo3 |", "|---|---:|---:|---:|---:|"]
    for tn in ("cards", "struct", "mana", "extra", "mull", "ref", "structure (struct+mana)", "extra+mull"):
        pps = []
        for k in ("ALL", "bo1", "bo3"):
            tp = t[k]["terms_pp"]
            pps.append(tp[tn] if tn in tp else (tp["struct"] + tp["mana"] if tn.startswith("structure") else tp["extra"] + tp["mull"]))
        L.append(f"| {tn} | " + " | ".join(_fmt_share(t[k]["terms"].get(tn)) for k in ("ALL", "bo1", "bo3")) + " | " + " / ".join(f"{x:.2f}" for x in pps) + " |")
    L += ["", "Cards (card-value terms) in pp of the self-preference scale: " + ", ".join(f"{k} {t[k]['cards_pp']:.2f}" for k in ("ALL", "bo1", "bo3")) + ".", ""]

    def tbl(title, key, groups, extra=""):
        L.extend([f"## {title}", "", "Share of T (cards term only is in these rows; they add to the cards share).", "", "| group | ALL | bo1 | bo3 |", "|---|---:|---:|---:|"])
        for g in groups:
            L.append(f"| {g} | " + " | ".join(_fmt_share(t[k][key].get(g)) for k in ("ALL", "bo1", "bo3")) + " |")
        L.append("")

    tbl("(b1) Cards term by rarity", "rarity", RARITY_GROUPS)
    tbl("(b2) Cards term by draft-exposure tertile (within set; T1 = least drafted)", "tertile", TERT_GROUPS)
    tbl("(b3) Cards term by presence (card in both decks / only in D_A / only in D_B)", "presence", PRES_GROUPS)
    L += ["## (b4) Cards term: beta disagreement vs castability-weight disagreement", "", "| piece | ALL | bo1 | bo3 |", "|---|---:|---:|---:|"]
    for lbl, key in (("beta disagreement x deck difference (all cards)", "cards_beta_part"), ("castability-weight disagreement (all cards)", "cards_w_part"),
                     ("beta disagreement, cards in both decks", "cards_beta_part_both"), ("weight disagreement, cards in both decks", "cards_w_part_both")):
        L.append(f"| {lbl} | " + " | ".join(_fmt_share(t[k].get(key)) for k in ("ALL", "bo1", "bo3")) + " |")
    L += ["", "## Per set x format: share of T from cards [95% CI] and from structure (struct+mana)", "", "| set | fmt | pools | cards | structure | mean T/2 pp |", "|---|---|---:|---:|---:|---:|"]
    for sc, d in res["terms_per_set"].items():
        for f in ("bo1", "bo3"):
            x = d.get(f)
            if x:
                L.append(f"| {sc} | {f} | {x['n_pools']} | {_fmt_share(x['terms']['cards'])} | {_fmt_share(x['terms']['structure (struct+mana)'])} | {x['mean_pref_pp']:.2f} |")
    # ---- stage
    st = res.get("stage") or {}
    if st.get("sets"):
        L += ["", "## (c) Draft prior vs Sealed adjustment (sets: " + ", ".join(st["sets"]) + ")", "",
              "Reconstruction of each half's final betas from a chain refit at the artifact's sigma / stacking weight (no cross-validation):", "",
              "| set | half | fmt | max abs err | rms err | rms artifact beta | lambda refit / artifact | stack w | refit s (setup + chain) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for sc, d in st["recon"].items():
            for h in ("A", "B"):
                for f, v in d[h].items():
                    L.append(f"| {sc} | {h} | {f} | {v['max_abs']:.1e} | {v['rms']:.1e} | {v['art_rms']:.3f} | "
                             f"{'-' if v['lam_rec'] is None else format(v['lam_rec'], '.4f')} / {'-' if v['lam_art'] is None else format(v['lam_art'], '.4f')} | {v['w']:g} | "
                             f"{d['seconds'][h][0]:.0f} + {d['seconds'][h][1]:.0f} |")
        for f in FMTS:
            p = st["pooled"].get(f)
            if not p:
                continue
            L += ["", f"### {f}: share of the cards term of T by chain piece (pooled over {', '.join(st['sets'])}; {p['n_pools']} pools)", "",
                  f"Cards term = {_fmt_share(p['cards_share_of_T'])} of T.", "", "| piece (share of the CARDS term would be piece / cards share) | share of T [95% CI] |", "|---|---:|"]
            for k, lbl in (("draft", "draft prior (lambda * D)"), ("draft_common", "  of which draft disagreement at common scale"),
                           ("draft_scale", "  of which lambda (scale) disagreement"), ("bo1_adj", "bo1 Sealed adjustment"),
                           ("bo3_adj", "bo3 Sealed adjustment"), ("sealed", "all Sealed adjustments"), ("resid", "reconstruction residual")):
                if "share_" + k in p:
                    L.append(f"| {lbl} | {_fmt_share(p['share_' + k])} |")
            for gk, ttl in (("by_rarity", "rarity"), ("by_tertile", "draft-exposure tertile")):
                L += ["", f"By {ttl}: share of T (draft | Sealed adjustments)", "", "| group | draft | Sealed |", "|---|---:|---:|"]
                for g, c in p[gk]["draft"].items():
                    L.append(f"| {g} | {_fmt_share(c)} | {_fmt_share(p[gk]['sealed'][g])} |")
            L += ["", f"beta_A - beta_B, weighted by |count difference| summed over pools ({f}); shares of the weighted E[d^2]; cov = 2 E[d_draft d_sealed]; noise scale = weighted RMS of lam_m sqrt(se_A^2 + se_B^2) with the draft-stage cluster-robust SEs (a rough sampling-noise yardstick for the draft part, not an exact variance)", "",
                  "| group | weight | RMS d | RMS draft | RMS Sealed | share draft | share Sealed | share cov | noise scale of draft part |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for g, m in p["var"].items():
                L.append(f"| {g} | {m['sum_w']:.0f} | {m['rms']:.3f} | {m['rms_draft']:.3f} | {m['rms_sealed']:.3f} | {pct(m['share_draft'])} | {pct(m['share_sealed'])} | {pct(m['share_cov'])} | {m.get('rms_noise', float('nan')):.3f} |")
        L += ["", "### Per set (cards term of T; weighted beta difference)", "", "| set | fmt | pools | draft share of cards T | Sealed share | RMS (beta_A - beta_B) | noise scale | var: draft / Sealed / cov | lambda A / B |", "|---|---|---:|---:|---:|---:|---:|---|---:|"]
        for sc, d in st["per_set_compact"].items():
            for f, x in d.items():
                L.append(f"| {sc} | {f} | {x['n_pools']} | {pct(x['share_draft'])} | {pct(x['share_sealed'])} | "
                         f"{'-' if x['rms_dbeta'] is None else format(x['rms_dbeta'], '.3f')} | {'-' if x['rms_noise'] is None else format(x['rms_noise'], '.3f')} | "
                         f"{pct(x['share_var_draft'])} / {pct(x['share_var_sealed'])} / {pct(x['share_var_cov'])} | {x['lam_A']:.3f} / {x['lam_B']:.3f} |")
    return "\n".join(L) + "\n"


# =====================================================================================================================
# structure / mana features: which of them carry the disagreement, and why        (`structure` subcommand)
# =====================================================================================================================
#
# Per feature j (raw units): b_Xj = theta_Xj / sd_Xj, S_X(D) = sum_j b_Xj (phi_Xj(D) - mu_Xj).  The mu terms are constant per model, so they cancel
# in T = [S_A(D_A) - S_A(D_B)] - [S_B(D_A) - S_B(D_B)] exactly, and T_j = b_Aj dphi_Aj - b_Bj dphi_Bj with dphi_Xj = phi_Xj(D_A) - phi_Xj(D_B)
# (phi differs between the models only in the dead-cards column: it uses each half's own keep rule).  card_T_contrib splits T_j into
# (b_A - b_B) * (dphi_A + dphi_B)/2 [coefficient disagreement] + (b_A + b_B)/2 * (dphi_A - dphi_B) [feature-value disagreement].

STRUCT_FEATURES = ["n_lands", "n_lands_dev2", "n_creatures", "n_noncreature", "curve_le1", "curve_2", "curve_3", "curve_4", "curve_5",
                   "curve_ge6", "mean_spell_mv", "n_main_colors", "n_splash_cards", "n_splash_pips"]
MANA_FEATURES = ["p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate", "dead_cards"]
FEATURES = STRUCT_FEATURES + MANA_FEATURES
NF = len(FEATURES)
FAMILIES = {
    "curve (mana-value bins, mean MV)": ["curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6", "mean_spell_mv"],
    "creature count": ["n_creatures", "n_noncreature"],
    "land count": ["n_lands", "n_lands_dev2"],
    "colours / splash": ["n_main_colors", "n_splash_cards", "n_splash_pips", "p_color_screw"],
    "mana-sim stats (screw, flood, mana spent, mulligan rate)": ["p_screw", "p_flood", "mana_spent", "mull_rate"],
    "dead cards": ["dead_cards"],
}
# n_lands + n_creatures + n_noncreature = n_cards and n_lands + sum(curve bins) = n_cards in every 40-card deck: two exact null directions
# of the design (broken only by the ridge), so single-coefficient shares inside this block depend on the ridge's gauge; only its sum does not.
GAUGE_BLOCK = ["n_lands", "n_creatures", "n_noncreature", "curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6"]
STRUCT_FMT_SETS = ("ALL", "bo1", "bo3")
RECON_TOL = 1e-5            # max |refit theta - artifact theta| (standardised units) for a refit to count as reproducing the artifact


def coarse_index() -> dict:
    """Groups that do not depend on how collinear coefficients share an effect: deck shape (everything about lands, creatures, curve, mana-sim),
    colours / splash, dead cards."""
    sh = [j for j, f in enumerate(FEATURES) if f not in ("n_main_colors", "n_splash_cards", "n_splash_pips", "p_color_screw", "dead_cards")]
    return {"deck shape (lands, creatures, curve, mana-sim stats)": np.array(sh), "colours / splash": np.array([FEATURES.index(f) for f in FAMILIES["colours / splash"]]),
            "dead cards": np.array([FEATURES.index("dead_cards")])}


def family_index() -> dict:
    return {fam: np.array([FEATURES.index(f) for f in fs], dtype=int) for fam, fs in FAMILIES.items()}


def align_features(names, values, fill: float = 0.0):
    """`values` (one per name of `names`) placed on FEATURES; `fill` where the model lacks the feature. Returns (vector (NF,), present (NF,) bool)."""
    out = np.full(NF, float(fill))
    present = np.zeros(NF, bool)
    for n, v in zip(names, np.asarray(values, float)):
        j = FEATURES.index(n)
        out[j], present[j] = v, True
    return out, present


def align_matrix(names, M):
    """Square matrix in the order of `names` placed on FEATURES (zeros for absent features)."""
    out = np.zeros((NF, NF))
    ix = np.array([FEATURES.index(n) for n in names], dtype=int)
    out[np.ix_(ix, ix)] = np.asarray(M, float)
    return out


def feature_T_parts(b_a, b_b, phi_aa, phi_ab, phi_ba, phi_bb):
    """Per-feature contribution to T of the structure + mana terms and its split (coefficient disagreement, feature-value disagreement).
    phi_xy = model x's feature values for deck y (x, y in A, B); b_x = raw-unit coefficients. total = beta_part + phi_part per feature, exactly."""
    return card_T_contrib(b_a, b_b, phi_aa, phi_ab, phi_ba, phi_bb)


def propagate_cov(data_cov, w_prior, scale, w, prev_cov):
    """Covariance (a block of coefficients) of a stage's deployed estimate incl. the uncertainty of its prior mean: the matrix version of
    cardvalue.propagate_se (whose diagonal it reproduces).  data_cov = cluster-robust covariance of the plain fit given the prior mean;
    w_prior = (se_lap / sigma)^2 per coefficient (the weight the prior receives); scale = lambda (1 if the prior is copied); w = stacking weight;
    prev_cov = covariance of the prior mean before scaling."""
    data_cov, prev_cov = np.asarray(data_cov, float), np.asarray(prev_cov, float)
    if w >= 1.0:
        g = np.asarray(w_prior, float)
        return data_cov + scale ** 2 * g[:, None] * prev_cov * g[None, :]
    g = 1.0 - w * (1.0 - np.asarray(w_prior, float))
    return w ** 2 * data_cov + scale ** 2 * g[:, None] * prev_cov * g[None, :]


def chi2_cells(d, var) -> dict:
    """z = d / sqrt(var) per entry and the sum of z^2 (df = number of entries)."""
    z = np.asarray(d, float) / np.sqrt(np.asarray(var, float))
    return dict(z=z, chi2=float((z ** 2).sum()), df=int(len(z)))


def mahalanobis(d, cov) -> tuple:
    """(d' cov^-1 d, df) for a difference vector d with covariance cov (positive definite)."""
    d = np.asarray(d, float)
    return float(d @ np.linalg.solve(np.asarray(cov, float), d)), int(len(d))


def gauge_null_vectors(idx) -> np.ndarray:
    """Raw-unit null directions of the design (columns, on the features `idx` of FEATURES): shifting the coefficients by c * u leaves every 40-card
    deck's prediction unchanged. u1 = n_lands + n_creatures + n_noncreature, u2 = n_lands + the six curve bins. Only those fully inside `idx` count."""
    idx = [int(i) for i in idx]
    pos = {j: i for i, j in enumerate(idx)}
    cols = []
    for names in (["n_lands", "n_creatures", "n_noncreature"], ["n_lands", "curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6"]):
        js = [FEATURES.index(n) for n in names]
        if all(j in pos for j in js):
            u = np.zeros(len(idx))
            u[[pos[j] for j in js]] = 1.0
            cols.append(u)
    return np.array(cols).T if cols else np.zeros((len(idx), 0))


def identifiable_mahalanobis(d, cov, idx, scale) -> tuple:
    """Mahalanobis distance of the difference d (raw units) restricted to `idx`, in the gauge-free subspace: the contrasts l'd with l'u = 0 for every
    null direction u (the ridge, not the data, fixes d along u, and a sandwich covariance has no variance there at all). `scale` (per feature, e.g. the
    mean sd of the two models) only conditions the numbers. Returns (chi2, df); df = len(idx) - number of null directions."""
    from scipy.linalg import null_space

    idx = np.asarray(idx, dtype=int)
    sc = np.asarray(scale, float)[idx]
    dt = np.asarray(d, float)[idx] * sc
    Ct = np.asarray(cov, float)[np.ix_(idx, idx)] * np.outer(sc, sc)
    U = gauge_null_vectors(idx) * sc[:, None]
    L = null_space(U.T) if U.shape[1] else np.eye(len(idx))
    return mahalanobis(L.T @ dt, L.T @ Ct @ L)


def raw_cov(cov_std, sd):
    sd = np.asarray(sd, float)
    return np.asarray(cov_std, float) / np.outer(sd, sd)


def cluster_cov(prob, coef, sig, group, w):
    """(H^-1, cluster-robust covariance) of every coefficient: the matrices behind glm.cluster_se (same sandwich, same G/(G-1) factor)."""
    import scipy.sparse as sp
    from scipy.linalg import cho_factor, cho_solve
    from scipy.special import expit

    from sealed_lab.model import glm

    if w is None:
        w = np.ones(prob.n)
    with glm._DENSE_LOCK:
        H = glm.hessian(prob, coef, sig, w)
        p = H.shape[0]
        try:
            Hinv = cho_solve(cho_factor(H, lower=True), np.eye(p))
        except np.linalg.LinAlgError:
            Hinv = np.linalg.pinv(H)
        G = int(group.max()) + 1
        r = w * (expit(prob.eta(coef)) - prob.y)
        n = prob.n
        blocks = [sp.csr_matrix((r, (group, np.arange(n))), shape=(G, n)) @ prob.C]
        if prob.D is not None:
            m = prob.D.shape[0]
            key = group.astype(np.int64) * m + prob.idx.astype(np.int64)
            uk, inv = np.unique(key, return_inverse=True)
            rsum = np.bincount(inv, weights=r, minlength=len(uk))
            pg, pd = (uk // m).astype(np.int64), (uk % m).astype(np.int64)
            blocks.append(sp.csr_matrix((rsum, (pg, np.arange(len(uk)))), shape=(G, len(uk))) @ prob.D[pd])
        S = sp.hstack(blocks, format="csr")
        cov = Hinv @ (S.T @ S).toarray() @ Hinv * (G / max(G - 1, 1))
    return Hinv, cov


# ---------------------------------------------------------------------------------------------------------------------
# step 1: per-pool, per-feature parts of T
# ---------------------------------------------------------------------------------------------------------------------

def model_ctx(m) -> dict:
    """Raw-unit coefficients of a model on FEATURES: names, sd, mu, present mask and b[fmt] = theta[0] / sd (draw 0)."""
    names = list(m.feature_names)
    sd, present = align_features(names, m.sd, fill=1.0)
    mu, _ = align_features(names, m.mu)
    b = {f: align_features(names, np.asarray(m._draws[f]["theta"][0], float) / m.sd)[0] for f in m.formats}
    return dict(names=names, sd=sd, mu=mu, present=present, b=b)


def structure_pool_record(row: dict, mA, mB, ctx: dict, trec: dict) -> dict:
    fmt = row["fmt"]
    DA = {int(k): int(v) for k, v in row["dirs"]["A"][VARIANT]["deck"].items()}
    DB = {int(k): int(v) for k, v in row["dirs"]["B"][VARIANT]["deck"].items()}
    phi = np.zeros((2, 2, NF))                          # [model A|B, deck D_A|D_B]
    sm = np.zeros((2, 2))
    for i, m in enumerate((mA, mB)):
        for j, D in enumerate((DA, DB)):
            _ref, _cards, st, ma, ph, _ids, _cnt = m._parts(D, fmt, False)
            phi[i, j] = align_features(m.feature_names, ph)[0]
            sm[i, j] = float(st[0] + ma[0])
    T_sm = term_T(sm[0, 0], sm[0, 1], sm[1, 0], sm[1, 1])
    tot, bpart, ppart = feature_T_parts(ctx["A"]["b"][fmt], ctx["B"]["b"][fmt], phi[0, 0], phi[0, 1], phi[1, 0], phi[1, 1])
    return dict(set=row["set"], fmt=fmt, pool=trec["pool"], T=trec["T"], s=trec["s"], T_sm=T_sm, T_sm_pickle=trec["terms"]["struct"] + trec["terms"]["mana"],
                tot=tot, bpart=bpart, ppart=ppart, phi=phi, chk_sum=float(abs(tot.sum() - T_sm)), chk_pickle=float(abs(T_sm - trec["terms"]["struct"] - trec["terms"]["mana"])))


def run_structure_pools(sh: Path, decomp: Path, out: Path, sets: list[str]) -> None:
    terms = load_terms(decomp, sets)
    rows_all = [json.loads(x) for x in (sh / "search_rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (out / "pools").mkdir(parents=True, exist_ok=True)
    for sc in sets:
        if sc not in terms:
            print(f"pools {sc}: no terms pickle in {decomp}, skipped")
            continue
        t0 = time.time()
        mA, mB = load_model(sh, sc, "A"), load_model(sh, sc, "B")
        ctx = {"A": model_ctx(mA), "B": model_ctx(mB)}
        by_pool = {r["pool"]: r for r in terms[sc]["records"]}
        recs = []
        with _home(home_of(sh, sc, "A")):
            for row in [r for r in rows_all if r["set"] == sc]:
                recs.append(structure_pool_record(row, mA, mB, ctx, by_pool[f'{row["set"]}|{row["fmt"]}|{row["draft_id"]}']))
        with open(out / "pools" / f"{sc}.pkl", "wb") as fh:
            pickle.dump(dict(set=sc, records=recs, ctx=ctx), fh)
        print(f"pools {sc}: {len(recs)} pools in {time.time() - t0:.0f}s; max|sum of features - struct+mana T| {max(r['chk_sum'] for r in recs):.1e}, "
              f"max|recomputed - stored struct+mana T| {max(r['chk_pickle'] for r in recs):.1e}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# step 2: chain refit with the covariance of the structure coefficients
# ---------------------------------------------------------------------------------------------------------------------

def refit_chain_cov(setup: HalfSetup, choose=None) -> dict:
    """`choose(st, name, prior, sigma, w, has_prior, spec) -> (spec, w)` (optional) replaces a stage's spec / stacking weight before its fit
    (scripts/theta_sigma_check.py: the structure SD chosen by CV). Default: the artifact's. refit_chain_fixed plus, per stage: the cluster-robust / Laplace covariance of the structure block, the prior weight of every structure
    coefficient (se_lap / sigma)^2, and the covariance of the DEPLOYED structure estimate incl. the uncertainty of its prior mean (`cov_total`,
    cardvalue.propagate_se in matrix form; its diagonal is checked against cardvalue.propagate_se itself)."""
    from types import SimpleNamespace

    from sealed_lab.model import cardvalue, glm, validate

    ps = setup.meta["prior_strength"]["stages"]
    prior_names = {n for n in ("draft", "bo1") if setup.n_games.get(n, 0) >= validate.MIN_PRIOR_GAMES}
    prior_ok = {n for n in setup.stages if n in prior_names}
    n_card, n_feat = setup.n_card, setup.n_feat
    p_deck = n_card + n_feat
    fits: dict = {}
    prev = None
    for name in [n for n in ("draft", "bo1", "bo3") if n in setup.stages]:
        st = setup.stages[name]
        lam = lam_raw = None
        if prev is None:
            prior = np.zeros(p_deck)
        elif prev["name"] == "draft":
            lam_raw, lam_se = cardvalue.fit_lambda(st, prev["deck"])
            lam = float(np.clip(lam_raw, *cardvalue.LAMBDA_CLIP))
            prior = lam * prev["deck"]
        else:
            prior = prev["deck"].copy()
        sigma, w = float(ps[name]["sigma"]), float(ps[name]["w"])
        spec = cardvalue.make_spec(st, n_card, n_feat, name)
        if choose is not None:
            spec, w = choose(st, name, prior, sigma, w, prev is not None, spec)
        coef_m, _, res = cardvalue.fit_stage(st, prior, spec, sigma, with_se=False)
        sigv = spec.vector(sigma)
        Hinv, cov = cluster_cov(st.prob, coef_m, sigv, st.group, st.w)
        se_lap, se_rob = np.sqrt(np.maximum(np.diag(Hinv), 0.0)), np.sqrt(np.maximum(np.diag(cov), 0.0))
        coef = glm.stack_coef(st.prob, coef_m, prior, w, spec.sig_cov, st.w) if w < 1.0 else coef_m
        k = st.prob.k
        t0, t1 = k + n_card, k + n_card + n_feat
        wp = (se_lap[t0:t1] / sigv[t0:t1]) ** 2
        data_cov = cov[t0:t1, t0:t1]
        f = SimpleNamespace(se=se_rob, se_lap=se_lap, k=k, spec=spec, sigma=sigma, lam=lam, w=w, beta_se_cond=None, beta_se=None, n_card=n_card)
        scale = lam if lam is not None else 1.0
        if prev is None:
            cov_tot = data_cov
            f.se_total = cardvalue.propagate_se(f, None)
        else:
            cov_tot = propagate_cov(data_cov, wp, scale, w, prev["cov_total"])
            f.se_total = cardvalue.propagate_se(f, prev["ns"])
        chk = float(np.abs(f.se_total[t0:t1] - np.sqrt(np.maximum(np.diag(cov_tot), 0.0))).max())
        fits[name] = dict(coef=coef, k=k, lam=lam, lam_raw=lam_raw, w=w, sigma=sigma, sig_theta=float(sigv[t0]), theta=coef[t0:t1].copy(),
                          cov_total=cov_tot, cov_data=data_cov, wp_theta=wp, se_lap_theta=se_lap[t0:t1], se_rob_theta=se_rob[t0:t1],
                          wp_cards_mean=float(np.mean((se_lap[k:k + n_card] / sigv[k:k + n_card]) ** 2)), se_cards_median=float(np.median(se_rob[k:k + n_card])),
                          check_propagate=chk, converged=bool(res.converged), n_games=int(st.n_games))
        if name in prior_ok:
            prev = dict(name=name, deck=coef[k:], ns=f, cov_total=cov_tot)
    fits["_prior_ok"] = sorted(prior_ok)
    fits["_formats"] = list(setup.meta["formats"])
    return fits


def structure_refit_half(sh: Path, sc: str, half: str, out: Path, choose=None) -> dict:
    t0 = time.time()
    ev = split_events_of(sh, sc)
    setup = HalfSetup(sc, home_of(sh, sc, half), ev[half], allow_plain=True)
    t_setup = time.time() - t0
    t1 = time.time()
    fits = refit_chain_cov(setup, choose)
    t_fit = time.time() - t1
    model = load_model(sh, sc, half)
    names = list(model.feature_names)
    info: dict = dict(set=sc, half=half, names=names, seconds_setup=t_setup, seconds_fit=t_fit, prior_ok=fits["_prior_ok"], formats=fits["_formats"],
                      sd_vs_meta=float(np.abs(setup.fspace.sd - model.sd).max()), mu_vs_meta=float(np.abs(setup.fspace.mu - model.mu).max()),
                      stages={}, recon={})
    arrays: dict = {}
    for name in [n for n in ("draft", "bo1", "bo3") if n in fits]:
        f = fits[name]
        arrays[f"{name}__theta"] = f["theta"]
        arrays[f"{name}__cov_total"] = f["cov_total"]
        arrays[f"{name}__cov_data"] = f["cov_data"]
        arrays[f"{name}__wp"] = f["wp_theta"]
        arrays[f"{name}__se_lap"] = f["se_lap_theta"]
        info["stages"][name] = {k: (None if f[k] is None else float(f[k])) for k in ("lam", "lam_raw", "w", "sigma", "sig_theta", "wp_cards_mean", "se_cards_median",
                                                                                   "check_propagate")}
        info["stages"][name].update(wp_theta_mean=float(f["wp_theta"].mean()), wp_theta_max=float(f["wp_theta"].max()), n_games=f["n_games"], converged=f["converged"],
                                    se_theta_median=float(np.median(f["se_rob_theta"])))
    for fm in fits["_formats"]:
        th_art = np.asarray(model._draws[fm]["theta"][0], float)
        info["recon"][fm] = dict(theta_max_abs=float(np.abs(fits[fm]["theta"] - th_art).max()), art_rms=float(np.sqrt(np.mean(th_art ** 2))))
    # the rebuild covers the plain chain only: a half whose artifact carries card-pair / archetype columns (synergy, archcond) does not reproduce
    info["ok"] = bool(all(v["theta_max_abs"] < RECON_TOL for v in info["recon"].values()))
    info["not_modelled"] = [k for k in ("synergy", "archcond", "attributes", "recency") if (model.meta.get(k) or {}).get("applied")]
    (out / "refit").mkdir(parents=True, exist_ok=True)
    np.savez(out / "refit" / f"{sc}_{half}.npz", **arrays)
    (out / "refit" / f"{sc}_{half}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"refit {sc} {half}: setup {t_setup:.0f}s, fit {t_fit:.0f}s; theta recon max abs "
          + ", ".join(f"{fm} {v['theta_max_abs']:.1e}" for fm, v in info["recon"].items())
          + f"; sd/mu vs meta {info['sd_vs_meta']:.1e}/{info['mu_vs_meta']:.1e}; propagate check "
          + ", ".join(f"{n} {v['check_propagate']:.1e}" for n, v in info["stages"].items()), flush=True)
    return info


# ---------------------------------------------------------------------------------------------------------------------
# step 3: report
# ---------------------------------------------------------------------------------------------------------------------

def load_structure_pools(out: Path, sets: list[str]) -> dict:
    res = {}
    for sc in sets:
        p = out / "pools" / f"{sc}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                res[sc] = pickle.load(fh)
    return res


def load_refit(out: Path, sc: str, half: str):
    p = out / "refit" / f"{sc}_{half}.npz"
    if not p.exists():
        return None
    info = json.loads((out / "refit" / f"{sc}_{half}.json").read_text(encoding="utf-8"))
    return dict(info=info, arr=dict(np.load(p)))


def usable_refits(refit: dict, sets: list[str]):
    """Sets whose two halves both reproduce their artifact's structure coefficients (RECON_TOL); returns (refit restricted to them, excluded {set: why})."""
    keep, bad = {}, {}
    for sc in sets:
        pair = [refit.get((sc, h)) for h in ("A", "B")]
        if any(x is None for x in pair):
            continue
        if all(x["info"]["ok"] for x in pair):
            keep.update({(sc, "A"): pair[0], (sc, "B"): pair[1]})
        else:
            bad[sc] = {h: dict(max_abs_theta_err=max(v["theta_max_abs"] for v in x["info"]["recon"].values()), not_modelled=x["info"]["not_modelled"])
                       for h, x in zip("AB", pair)}
    return keep, bad


def summarise_structure(recs: list[dict]) -> dict:
    if not recs:
        return {}
    fam = family_index()
    COARSE = coarse_index()
    gix = np.array([FEATURES.index(f) for f in GAUGE_BLOCK], dtype=int)
    out: dict = dict(n_pools=len(recs))
    out["structure"] = _cell(recs, lambda r: r["T_sm"])
    out["feature"] = {f: _cell(recs, lambda r, j=j: r["tot"][j]) for j, f in enumerate(FEATURES)}
    out["family"] = {fm: _cell(recs, lambda r, ix=ix: r["tot"][ix].sum()) for fm, ix in fam.items()}
    out["coarse"] = {nm: _cell(recs, lambda r, ix=ix: r["tot"][ix].sum()) for nm, ix in COARSE.items()}
    out["coarse_pp"] = {nm: float(np.mean([50 * r["s"] * r["tot"][ix].sum() for r in recs])) for nm, ix in COARSE.items()}
    out["gauge_block"] = _cell(recs, lambda r: r["tot"][gix].sum())
    out["gauge_free_rest"] = _cell(recs, lambda r: r["T_sm"] - r["tot"][gix].sum())
    out["coef_part"] = _cell(recs, lambda r: r["bpart"].sum())
    out["phi_part"] = _cell(recs, lambda r: r["ppart"].sum())
    out["family_coef_part"] = {fm: _cell(recs, lambda r, ix=ix: r["bpart"][ix].sum()) for fm, ix in fam.items()}
    out["feature_pp"] = {f: float(np.mean([50 * r["s"] * r["tot"][j] for r in recs])) for j, f in enumerate(FEATURES)}
    out["family_pp"] = {fm: float(np.mean([50 * r["s"] * r["tot"][ix].sum() for r in recs])) for fm, ix in fam.items()}
    out["structure_pp"] = float(np.mean([50 * r["s"] * r["T_sm"] for r in recs]))
    out["frac_pools_positive"] = {f: float(np.mean([r["tot"][j] > 0 for r in recs])) for j, f in enumerate(FEATURES)}
    # how different the two builds are: |dphi| in raw units (average of the two models' views) and in sd units
    sds = np.array([r["phi"] for r in recs])
    dphi = 0.5 * ((sds[:, 0, 0] - sds[:, 0, 1]) + (sds[:, 1, 0] - sds[:, 1, 1]))            # (n, NF)
    out["abs_dphi"] = {f: float(np.mean(np.abs(dphi[:, j]))) for j, f in enumerate(FEATURES)}
    out["frac_dphi_nonzero"] = {f: float(np.mean(np.abs(dphi[:, j]) > 1e-9)) for j, f in enumerate(FEATURES)}
    out["mean_phi_own"] = {f: float(np.mean(0.5 * (sds[:, 0, 0, j] + sds[:, 1, 1, j]))) for j, f in enumerate(FEATURES)}
    return out


def _stage_cells(sp: dict, refit: dict, sets: list[str]):
    """Per (set, fmt) cell with refits for both halves: raw coefficient vectors / covariances / stage pieces on FEATURES."""
    cells = []
    for sc in sets:
        if sc not in sp:
            continue
        rA, rB = refit.get((sc, "A")), refit.get((sc, "B"))
        if rA is None or rB is None:
            continue
        ctx = sp[sc]["ctx"]
        side = {}
        for h, rf in (("A", rA), ("B", rB)):
            names, arr, info = rf["info"]["names"], rf["arr"], rf["info"]
            c = ctx[h]
            lam1 = (info["stages"].get("bo1") or {}).get("lam")
            D = align_features(names, arr["draft__theta"])[0] if "draft__theta" in arr else np.zeros(NF)
            th = {f: align_features(names, arr[f"{f}__theta"])[0] for f in info["formats"]}
            parts = chain_parts(D, lam1 if lam1 is not None else 1.0, th.get("bo1"), th.get("bo3"), "bo1" in info["prior_ok"],
                                (info["stages"].get("bo3") or {}).get("lam"))
            side[h] = dict(c=c, info=info, arr=arr, names=names, parts=parts)
        for fm in rA["info"]["formats"]:
            if fm not in rB["info"]["formats"]:
                continue
            cell = dict(set=sc, fmt=fm)
            for h in ("A", "B"):
                s_ = side[h]
                sd = s_["c"]["sd"]
                cov_std = align_matrix(s_["names"], s_["arr"][f"{fm}__cov_total"])
                cell[h] = dict(b=s_["c"]["b"][fm], sd=sd, present=s_["c"]["present"], cov=raw_cov(cov_std, sd),
                               pieces={nm: v / sd for nm, v in s_["parts"].get(fm, {}).items()}, stages=s_["info"]["stages"])
                cell[h]["pieces"]["resid"] = cell[h]["b"] - sum(cell[h]["pieces"].values()) if cell[h]["pieces"] else np.zeros(NF)
            cells.append(cell)
    return cells


def noise_tables(cells: list[dict]) -> dict:
    """z = (b_A - b_B) / sqrt(se_A^2 + se_B^2) per (set, fmt, feature), chi-square per family (diagonal) and the full-covariance Mahalanobis
    distance of the whole structure block per cell."""
    fam = family_index()
    rows = []
    for c in cells:
        both = c["A"]["present"] & c["B"]["present"]
        d = c["A"]["b"] - c["B"]["b"]
        var = np.diag(c["A"]["cov"]) + np.diag(c["B"]["cov"])
        z = np.where(both, d / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
        ix = np.nonzero(both)[0]
        cov = c["A"]["cov"] + c["B"]["cov"]
        sdm = 0.5 * (c["A"]["sd"] + c["B"]["sd"])
        gb = [j for j in ix if FEATURES[j] in GAUGE_BLOCK]
        rest = [j for j in ix if FEATURES[j] not in GAUGE_BLOCK]
        m2, df = identifiable_mahalanobis(d, cov, ix, sdm)
        mg, dfg = identifiable_mahalanobis(d, cov, gb, sdm)
        mr, dfr = identifiable_mahalanobis(d, cov, rest, sdm)
        m2_raw, df_raw = mahalanobis(d[ix], cov[np.ix_(ix, ix)])
        rows.append(dict(set=c["set"], fmt=c["fmt"], z=z, both=both, mahal=m2, mahal_df=df, mahal_gauge=mg, mahal_gauge_df=dfg, mahal_rest=mr, mahal_rest_df=dfr,
                         mahal_ungauged=m2_raw, mahal_ungauged_df=df_raw))
    res: dict = dict(cells=rows, family={}, feature={})
    for sel in STRUCT_FMT_SETS:
        rs = [r for r in rows if sel == "ALL" or r["fmt"] == sel]
        if not rs:
            continue
        res["family"][sel] = {}
        for fm, ix in fam.items():
            chi = float(sum(r["z"][j] ** 2 for r in rs for j in ix if r["both"][j]))
            df = int(sum(1 for r in rs for j in ix if r["both"][j]))
            res["family"][sel][fm] = dict(chi2=chi, df=df, ratio=chi / df if df else float("nan"), p=_chi2_sf(chi, df))
        for lbl, a, b in (("gauge block (9 coefficients, 7 identifiable contrasts), full covariance", "mahal_gauge", "mahal_gauge_df"),
                          ("the other features, full covariance", "mahal_rest", "mahal_rest_df"),
                          ("ALL features, gauge-free full covariance (Mahalanobis)", "mahal", "mahal_df"),
                          ("(for contrast) ALL features, full covariance WITHOUT removing the gauge", "mahal_ungauged", "mahal_ungauged_df")):
            chi = float(sum(r[a] for r in rs))
            df = int(sum(r[b] for r in rs))
            res["family"][sel][lbl] = dict(chi2=chi, df=df, ratio=chi / df, p=_chi2_sf(chi, df))
        chi = float(sum((r["z"][r["both"]] ** 2).sum() for r in rs))
        df = int(sum(r["both"].sum() for r in rs))
        res["family"][sel]["ALL features, diagonal"] = dict(chi2=chi, df=df, ratio=chi / df, p=_chi2_sf(chi, df))
        res["feature"][sel] = {}
        for j, f in enumerate(FEATURES):
            zz = np.array([r["z"][j] for r in rs if r["both"][j]])
            if len(zz):
                res["feature"][sel][f] = dict(n=int(len(zz)), mean_z2=float((zz ** 2).mean()), max_abs_z=float(np.abs(zz).max()))
    return res


def _chi2_sf(x, df):
    from scipy.stats import chi2

    return float(chi2.sf(x, df)) if df > 0 else float("nan")


def pool_noise(recs: list[dict], cells: list[dict]) -> dict:
    """Sampling-noise SD of the structure part of T per pool: var = dphi_A' C_A dphi_A + dphi_B' C_B dphi_B with the raw-unit covariance of each
    model's coefficients (conditional on the decks, the coefficients being the only random thing), vs the observed structure part of T."""
    cmap = {(c["set"], c["fmt"]): c for c in cells}
    sd_l, sd_pp, obs_l, obs_pp, s_all = [], [], [], [], []
    for r in recs:
        c = cmap.get((r["set"], r["fmt"]))
        if c is None:
            continue
        dA, dB = r["phi"][0, 0] - r["phi"][0, 1], r["phi"][1, 0] - r["phi"][1, 1]
        v = float(dA @ c["A"]["cov"] @ dA + dB @ c["B"]["cov"] @ dB)
        sd_l.append(np.sqrt(max(v, 0.0)))
        sd_pp.append(50 * r["s"] * np.sqrt(max(v, 0.0)))
        obs_l.append(r["T_sm"])
        obs_pp.append(50 * r["s"] * r["T_sm"])
    if not sd_l:
        return {}
    return dict(n_pools=len(sd_l), mean_noise_sd_logit=float(np.mean(sd_l)), mean_noise_sd_pp=float(np.mean(sd_pp)), mean_T_structure_logit=float(np.mean(obs_l)),
                mean_T_structure_pp=float(np.mean(obs_pp)), rms_T_structure_logit=float(np.sqrt(np.mean(np.square(obs_l)))),
                rms_noise_logit=float(np.sqrt(np.mean(np.square(sd_l)))))


def stage_piece_tables(recs_by_set: dict, cells: list[dict]) -> dict:
    """Pooled share of T (cards + all) coming from each chain piece of the structure coefficients, per family; and RMS coefficient differences."""
    fam = family_index()
    cmap = {(c["set"], c["fmt"]): c for c in cells}
    out: dict = {}
    for fm in FMTS:
        rows = []
        for sc, recs in recs_by_set.items():
            for r in recs:
                if r["fmt"] != fm or (sc, fm) not in cmap:
                    continue
                c = cmap[(sc, fm)]
                row = dict(pool=r["pool"], T=r["T"], T_sm=r["T_sm"])
                piece_tot = {}
                for nm in c["A"]["pieces"]:
                    if nm not in c["B"]["pieces"]:
                        continue
                    t, _, _ = feature_T_parts(c["A"]["pieces"][nm], c["B"]["pieces"][nm], r["phi"][0, 0], r["phi"][0, 1], r["phi"][1, 0], r["phi"][1, 1])
                    piece_tot[nm] = t
                row["pieces"] = piece_tot
                rows.append(row)
        if not rows:
            continue
        names = sorted({nm for r in rows for nm in r["pieces"]})
        o: dict = dict(n_pools=len(rows), structure=_cell(rows, lambda r: r["T_sm"]))
        o["piece"] = {nm: _cell(rows, lambda r, nm=nm: r["pieces"].get(nm, np.zeros(NF)).sum()) for nm in names}
        o["sealed"] = _cell(rows, lambda r: sum(v.sum() for k, v in r["pieces"].items() if k in ("bo1_adj", "bo3_adj")))
        o["family"] = {fam_n: {nm: _cell(rows, lambda r, nm=nm, ix=ix: r["pieces"].get(nm, np.zeros(NF))[ix].sum()) for nm in names}
                       for fam_n, ix in fam.items()}
        for fam_n, ix in fam.items():
            o["family"][fam_n]["sealed"] = _cell(rows, lambda r, ix=ix: sum(v[ix].sum() for k, v in r["pieces"].items() if k in ("bo1_adj", "bo3_adj")))
        out[fm] = o
    # RMS of the coefficient differences (std units: raw difference x mean sd of the two models) by piece, per family, over (set, fmt) cells
    rms: dict = {}
    for fm in FMTS:
        cs = [c for c in cells if c["fmt"] == fm]
        if not cs:
            continue
        rms[fm] = {}
        for fam_n, ix in list(fam.items()) + [("ALL", np.arange(NF))]:
            acc: dict = {}
            for c in cs:
                both = c["A"]["present"] & c["B"]["present"]
                sdm = 0.5 * (c["A"]["sd"] + c["B"]["sd"])
                use = [j for j in ix if both[j]]
                if not use:
                    continue
                d_tot = (c["A"]["b"] - c["B"]["b"])[use] * sdm[use]
                acc.setdefault("total", []).append(d_tot ** 2)
                if c["A"]["pieces"] and c["B"]["pieces"]:
                    dr = (c["A"]["pieces"].get("draft", np.zeros(NF)) - c["B"]["pieces"].get("draft", np.zeros(NF)))[use] * sdm[use]
                    ds = sum((c["A"]["pieces"].get(k, np.zeros(NF)) - c["B"]["pieces"].get(k, np.zeros(NF)))[use] * sdm[use] for k in ("bo1_adj", "bo3_adj"))
                    acc.setdefault("draft", []).append(dr ** 2)
                    acc.setdefault("sealed", []).append(ds ** 2 if not np.isscalar(ds) else np.zeros(len(use)))
            rms[fm][fam_n] = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in acc.items()}
    out["rms_dcoef_std"] = rms
    return out


def prior_weight_tables(refit: dict, sets: list[str]) -> dict:
    """Per stage: the prior SDs actually applied to cards and to the structure block, and the weight the prior gets on the structure block."""
    rows = []
    for sc in sets:
        for h in ("A", "B"):
            rf = refit.get((sc, h))
            if rf is None:
                continue
            for name, s in rf["info"]["stages"].items():
                rows.append(dict(set=sc, half=h, stage=name, n_games=s["n_games"], sigma_card=s["sigma"], sigma_theta=s["sig_theta"], w_stack=s["w"], lam=s["lam"],
                                 wp_theta_mean=s["wp_theta_mean"], wp_theta_max=s["wp_theta_max"], wp_cards_mean=s["wp_cards_mean"],
                                 se_theta_median=s["se_theta_median"], se_cards_median=s["se_cards_median"]))
    return dict(rows=rows)


def mx_assessment(sp: dict, refit: dict, sets: list[str]) -> dict:
    """What could a cross-set prior on the structure block do?  tau_j = between-set SD of the sets' structure effects (raw units; the mean over the
    formats of the final draw-0 theta / sd, as crossset.other_set_effects builds it, from the half models; the gauge block's tau is inflated by the ridge
    gauge) against the SE of the same coefficient (mean over the sets' formats of the final-estimate SE).  tau_noise_free^2 = max(0, tau^2 - mean SE^2)
    is what is left of the between-set spread once the estimates' own sampling noise is taken out.  Prior weights (the weight the prior mean gets,
    (Laplace SE / prior SD)^2): default chain (SD = sigma_theta), MX as built (SD = max(sigma_theta, tau)), idealised MX (SD = tau)."""
    eff, se2 = {}, {}
    for sc in sets:
        for h in ("A", "B"):
            if sc not in sp or refit.get((sc, h)) is None:
                continue
            c = sp[sc]["ctx"][h]
            rf = refit[(sc, h)]
            names, arr = rf["info"]["names"], rf["arr"]
            eff[(sc, h)] = np.where(c["present"], np.mean([c["b"][f] for f in c["b"]], axis=0), np.nan)
            v = np.mean([align_features(names, np.diag(arr[f"{f}__cov_total"]))[0] / c["sd"] ** 2 for f in c["b"]], axis=0)          # raw-unit variance
            se2[(sc, h)] = np.where(c["present"], v, np.nan)
    tau = {}
    for h in ("A", "B"):
        M = np.array([eff[(sc, h)] for sc in sets if (sc, h) in eff])
        with np.errstate(all="ignore"):
            tau[h] = np.nanstd(M, axis=0, ddof=1)
    tau_raw = np.nanmean([tau["A"], tau["B"]], axis=0)
    mean_se2 = np.nanmean([se2[k] for k in se2], axis=0)
    tau_free = np.sqrt(np.maximum(tau_raw ** 2 - mean_se2, 0.0))
    rows = []
    for sc in sets:
        for h in ("A", "B"):
            rf = refit.get((sc, h))
            if rf is None or (sc, h) not in eff:
                continue
            names, arr, info = rf["info"]["names"], rf["arr"], rf["info"]
            first = "draft" if "draft__theta" in arr else "bo1"
            sd = sp[sc]["ctx"][h]["sd"]
            se_data = align_features(names, np.sqrt(np.diag(arr[f"{first}__cov_data"])))[0]
            se_lap = align_features(names, arr[f"{first}__se_lap"])[0]
            present = sp[sc]["ctx"][h]["present"]
            sig = info["stages"][first]["sig_theta"]
            tau_std = tau_raw * sd
            sig_eff = np.maximum(sig, np.nan_to_num(tau_std))
            rows.append(dict(set=sc, half=h, first_stage=first, sig_theta=sig, se_data=se_data, tau_std=tau_std, tau_free_std=tau_free * sd,
                             w_default=(se_lap / sig) ** 2, w_as_built=(se_lap / sig_eff) ** 2, w_ideal=se_lap ** 2 / (se_lap ** 2 + tau_std ** 2), present=present))
    out: dict = dict(tau_raw={f: float(tau_raw[j]) for j, f in enumerate(FEATURES)}, tau_noise_free_raw={f: float(tau_free[j]) for j, f in enumerate(FEATURES)},
                     rows=[], by_family={})
    for r in rows:
        pr = r["present"]
        out["rows"].append(dict(set=r["set"], half=r["half"], first_stage=r["first_stage"], sig_theta=r["sig_theta"], median_se=float(np.median(r["se_data"][pr])),
                                median_tau=float(np.nanmedian(r["tau_std"][pr])), median_w_default=float(np.median(r["w_default"][pr])),
                                median_w_as_built=float(np.nanmedian(r["w_as_built"][pr])), median_w_ideal=float(np.nanmedian(r["w_ideal"][pr]))))
    groups = {fam_n: [j for j in ix if FEATURES[j] not in GAUGE_BLOCK] for fam_n, ix in family_index().items()}
    groups["gauge block (tau inflated by the ridge gauge)"] = [FEATURES.index(f) for f in GAUGE_BLOCK]
    groups["all non-gauge features"] = [j for j in range(NF) if FEATURES[j] not in GAUGE_BLOCK]
    for gname, ix in groups.items():
        acc = {k: [] for k in ("se_data", "tau_std", "tau_free_std", "w_default", "w_as_built", "w_ideal")}
        for r in rows:
            for j in ix:
                if r["present"][j] and np.isfinite(r["tau_std"][j]):
                    for k in acc:
                        acc[k].append(r[k][j])
        out["by_family"][gname] = {k: float(np.median(v)) for k, v in acc.items() if v}
        out["by_family"][gname]["n"] = len(acc["se_data"])
    return out


def run_structure_report(sh: Path, out: Path, sets: list[str], stage_sets: list[str]) -> str:
    sp = load_structure_pools(out, sets)
    sets = [s for s in sets if s in sp]
    recs_all = [r for s in sets for r in sp[s]["records"]]
    refit = {(s, h): load_refit(out, s, h) for s in sets for h in ("A", "B")}
    refit = {k: v for k, v in refit.items() if v is not None}
    refit, excluded = usable_refits(refit, sets)
    res: dict = dict(sets=sets, n_pools=len(recs_all), n_boot=N_BOOT, features=FEATURES, families={k: v for k, v in FAMILIES.items()}, gauge_block=GAUGE_BLOCK)
    res["checks"] = dict(max_sum_of_features_vs_struct_mana_T=max(r["chk_sum"] for r in recs_all), max_recomputed_vs_stored_struct_mana_T=max(r["chk_pickle"] for r in recs_all),
                         max_total_check_in_family_sums=max(abs(sum(r["tot"][ix].sum() for ix in family_index().values()) - r["T_sm"]) for r in recs_all))
    res["summary"] = {name: summarise_structure(rs) for name, rs in selections(recs_all).items()}
    res["per_set"] = {s: {f: summarise_structure([r for r in sp[s]["records"] if f == "ALL" or r["fmt"] == f]) for f in ("ALL", "bo1", "bo3")} for s in sets}
    for s in res["per_set"]:                         # compact per-set json
        for f in res["per_set"][s]:
            d = res["per_set"][s][f]
            res["per_set"][s][f] = dict(n_pools=d["n_pools"], structure=d["structure"], family=d["family"], family_pp=d["family_pp"])
    cells = _stage_cells(sp, refit, sets)
    res["refit_sets"] = sorted({c["set"] for c in cells})
    res["refit_excluded"] = excluded
    res["noise"] = noise_tables(cells)
    for c in res["noise"]["cells"]:
        c["z"] = [None if not b else float(x) for x, b in zip(c["z"], c["both"])]
        c["both"] = [bool(x) for x in c["both"]]
    res["pool_noise"] = {}
    for sel in STRUCT_FMT_SETS:
        res["pool_noise"][sel] = pool_noise([r for r in recs_all if sel == "ALL" or r["fmt"] == sel], cells)
    sset = [s for s in stage_sets if s in res["refit_sets"]]
    res["stage_sets_requested"] = sset
    res["stage"] = {"requested_sets": stage_piece_tables({s: sp[s]["records"] for s in sset}, [c for c in cells if c["set"] in sset]),
                    "all_refit_sets": stage_piece_tables({s: sp[s]["records"] for s in res["refit_sets"]}, cells)}
    res["prior_weights"] = prior_weight_tables(refit, sets)
    res["mx"] = mx_assessment(sp, refit, res["refit_sets"])
    res["refit_checks"] = dict(
        max_theta_recon_abs=max(v["theta_max_abs"] for rf in refit.values() for v in rf["info"]["recon"].values()),
        max_sd_vs_meta=max(rf["info"]["sd_vs_meta"] for rf in refit.values()), max_propagate_check=max(s["check_propagate"] for rf in refit.values() for s in rf["info"]["stages"].values()),
        seconds={f"{s}_{h}": [rf["info"]["seconds_setup"], rf["info"]["seconds_fit"]] for (s, h), rf in refit.items()})
    (out / "structure.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    md = structure_markdown(res)
    extra = out / "interpretation.md"
    if extra.exists():
        md += "\n" + extra.read_text(encoding="utf-8")
    (out / "structure.md").write_text(md, encoding="utf-8")
    return md


def structure_markdown(res: dict) -> str:
    S = res["summary"]
    L = ["# Which structure / mana features carry the split-half disagreement", "",
         f"{res['n_pools']} pool x format cells, {len(res['sets'])} sets, draw 0, d0_fast top decks (see decompose.md for the definition of T). Shares = sum over pools of the "
         f"component / sum of T; 95% cluster bootstrap over pools ({res['n_boot']} resamples). pp = 100 s x / 2 (self-preference scale).", ""]
    ck = res["checks"]
    L += [f"Sum check: max |sum of per-feature parts - struct+mana part of T| = {ck['max_sum_of_features_vs_struct_mana_T']:.1e} over all pools "
          f"(the mu terms cancel exactly); max |recomputed - stored struct+mana T| = {ck['max_recomputed_vs_stored_struct_mana_T']:.1e}; "
          f"max |family sums - struct+mana T| = {ck['max_total_check_in_family_sums']:.1e}.", ""]
    L += ["## 1a. Families (share of the whole T)", "", "| family | ALL | bo1 | bo3 | pp ALL / bo1 / bo3 |", "|---|---:|---:|---:|---:|"]
    for fm in list(FAMILIES) + ["structure (struct+mana) total"]:
        cells = [S[k]["family"][fm] if fm in FAMILIES else S[k]["structure"] for k in STRUCT_FMT_SETS]
        pps = [S[k]["family_pp"][fm] if fm in FAMILIES else S[k]["structure_pp"] for k in STRUCT_FMT_SETS]
        L.append(f"| {fm} | " + " | ".join(_fmt_share(c) for c in cells) + " | " + " / ".join(f"{x:.2f}" for x in pps) + " |")
    L += ["", "Gauge-robust grouping (the shares inside 'deck shape' trade off against each other, e.g. n_lands vs p_screw vs p_flood):", "",
          "| group | ALL | bo1 | bo3 | pp ALL / bo1 / bo3 |", "|---|---:|---:|---:|---:|"]
    for nm in coarse_index():
        L.append(f"| {nm} | " + " | ".join(_fmt_share(S[k]["coarse"][nm]) for k in STRUCT_FMT_SETS) + " | " + " / ".join(f"{S[k]['coarse_pp'][nm]:.2f}" for k in STRUCT_FMT_SETS) + " |")
    L += ["", "Gauge: n_lands + n_creatures + n_noncreature = 40 and n_lands + sum(curve bins) = 40 in every deck, so the ridge, not the data, fixes how "
          "the effect is shared among those 9 coefficients; only their SUM is meaningful:", "",
          "| block | ALL | bo1 | bo3 |", "|---|---:|---:|---:|",
          "| lands + creatures + non-creatures + 6 curve bins (gauge block) | " + " | ".join(_fmt_share(S[k]["gauge_block"]) for k in STRUCT_FMT_SETS) + " |",
          "| the rest (colours / splash, mean MV, land deviation, mana-sim stats, dead cards) | " + " | ".join(_fmt_share(S[k]["gauge_free_rest"]) for k in STRUCT_FMT_SETS) + " |", ""]
    L += ["## 1b. Per feature (share of T); sorted by ALL", "",
          "| feature | family | ALL | bo1 | bo3 | pp ALL | share of pools with T_j > 0 |", "|---|---|---:|---:|---:|---:|---:|"]
    fam_of = {f: fm for fm, fs in FAMILIES.items() for f in fs}
    order = sorted(FEATURES, key=lambda f: -abs(S["ALL"]["feature"][f]["share"]))
    for f in order:
        L.append(f"| {f}{' *' if f in GAUGE_BLOCK else ''} | {fam_of[f].split(' (')[0]} | " + " | ".join(_fmt_share(S[k]["feature"][f]) for k in STRUCT_FMT_SETS)
                 + f" | {S['ALL']['feature_pp'][f]:.3f} | {pct(S['ALL']['frac_pools_positive'][f])} |")
    L += ["", "(* = member of the gauge block.)  The whole struct+mana part splits into coefficient disagreement "
          f"{_fmt_share(S['ALL']['coef_part'])} and feature-value disagreement {_fmt_share(S['ALL']['phi_part'])} of T (the latter is the dead-cards column: "
          "each half computes castability with its own keep rule).", ""]
    L += ["## 5. How different are the two models' builds (D_A vs D_B)", "",
          "| feature | mean abs(dphi) raw | share of pools with dphi != 0 | mean phi of the own deck |", "|---|---:|---:|---:|"]
    for f in order[:10]:
        L.append(f"| {f} | {S['ALL']['abs_dphi'][f]:.3f} | {pct(S['ALL']['frac_dphi_nonzero'][f])} | {S['ALL']['mean_phi_own'][f]:.3f} |")
    L += ["", "bo1 vs bo3 (mean abs dphi, raw): " + "; ".join(f"{f} {S['bo1']['abs_dphi'][f]:.3f} / {S['bo3']['abs_dphi'][f]:.3f}" for f in order[:6]), ""]
    # ---- noise
    nz = res["noise"]
    L += ["## 2. Is the coefficient disagreement just sampling noise?", "",
          f"z = (b_A - b_B) / sqrt(se_A^2 + se_B^2), se from a chain refit of each half at its own sigma / stacking weight (no CV) incl. the propagated uncertainty of "
          f"the prior mean (cardvalue.propagate_se; a matrix version gives the covariances). Sets with refits that reproduce the artifacts: {', '.join(res['refit_sets'])}"
          + (f" (excluded: {', '.join(f'{k} ({v})' for k, v in res['refit_excluded'].items())})" if res["refit_excluded"] else "") + f". chi-square = sum of z^2 over "
          "(set x format x feature in the family); under pure sampling noise chi2/df = 1. 'Full covariance' uses the whole structure block covariance of both halves "
          "(gauge-free: the two null directions of the design are projected out before testing); the diagonal version ignores the correlations (and counts the null-direction "
          "coefficients individually, with a sandwich SE that has no variance along the null directions).", "",
          "| family | ALL chi2 / df | ratio | p | bo1 ratio | bo3 ratio |", "|---|---:|---:|---:|---:|---:|"]
    fams = list(FAMILIES) + ["ALL features, diagonal"] + [k for k in nz["family"]["ALL"] if k not in FAMILIES and k != "ALL features, diagonal"]
    for fm in fams:
        a, b1, b3 = nz["family"]["ALL"].get(fm), nz["family"].get("bo1", {}).get(fm), nz["family"].get("bo3", {}).get(fm)
        if a:
            L.append(f"| {fm} | {a['chi2']:.0f} / {a['df']} | {a['ratio']:.2f} | {a['p']:.2g} | {b1['ratio']:.2f} | {b3['ratio']:.2f} |")
    L += ["", "Per feature: mean z^2 over cells (ALL) [max |z|]: " + ", ".join(f"{f} {v['mean_z2']:.1f} [{v['max_abs_z']:.1f}]" for f, v in nz["feature"]["ALL"].items()), ""]
    L += ["Per-cell gauge-free Mahalanobis chi2/df (full covariance); when both halves of a set have stacking weight 0 in bo3, bo3 = bo1 and the two cells are the same number:", "",
          "| set | bo1 | bo3 |", "|---|---:|---:|"]
    byset: dict = {}
    for c in nz["cells"]:
        byset.setdefault(c["set"], {})[c["fmt"]] = c["mahal"] / c["mahal_df"]
    for s, d in byset.items():
        L.append(f"| {s} | {d.get('bo1', float('nan')):.2f} | {d.get('bo3', float('nan')):.2f} |")
    L += ["", "Noise on the structure part of T itself (conditional on the two decks; coefficient errors only):", "",
          "| selection | pools | mean noise SD of T_struct (logit) | mean observed T_struct (logit) | RMS observed | RMS noise SD | noise SD pp | observed pp |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in STRUCT_FMT_SETS:
        p = res["pool_noise"].get(k)
        if p:
            L.append(f"| {k} | {p['n_pools']} | {p['mean_noise_sd_logit']:.4f} | {p['mean_T_structure_logit']:.4f} | {p['rms_T_structure_logit']:.4f} | {p['rms_noise_logit']:.4f} | "
                     f"{p['mean_noise_sd_pp']:.3f} | {p['mean_T_structure_pp']:.3f} |")
    # ---- stage
    L += ["", "## 3. Draft prior vs Sealed adjustment of the structure coefficients", "",
          "Prior SDs applied (structure columns: sigma_theta = clip(4 x card sigma, lo, hi) with (lo, hi) = (0.2, 1.0) draft, (0.05, 0.5) bo1 / bo3, in standardised units; cards: the card sigma) and "
          "the weight the prior gets, (se_lap / sigma)^2 (1 = coefficient fixed at its prior mean, 0 = data only):", "",
          "| set | half | stage | games | card sigma | theta sigma | stack w | lambda | prior weight, theta (mean / max) | prior weight, cards (mean) | median SE theta | median SE card |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["prior_weights"]["rows"]:
        L.append(f"| {r['set']} | {r['half']} | {r['stage']} | {r['n_games']} | {r['sigma_card']:.4f} | {r['sigma_theta']:.3f} | {r['w_stack']:g} | "
                 f"{'-' if r['lam'] is None else format(r['lam'], '.3f')} | {r['wp_theta_mean']:.3f} / {r['wp_theta_max']:.3f} | {r['wp_cards_mean']:.3f} | "
                 f"{r['se_theta_median']:.3f} | {r['se_cards_median']:.3f} |")
    st = res["stage"]
    for key, ttl in (("requested_sets", f"sets {', '.join(res['stage_sets_requested'])}"), ("all_refit_sets", f"all sets with a refit ({', '.join(res['refit_sets'])})")):
        d = st[key]
        for fm in FMTS:
            if fm not in d:
                continue
            o = d[fm]
            L += ["", f"### {fm}: share of T by chain piece of the structure coefficients ({ttl}; {o['n_pools']} pools)", "",
                  f"Structure (struct+mana) = {_fmt_share(o['structure'])} of T.", "", "| piece | share of T |", "|---|---:|"]
            for nm, lbl in (("draft", "draft prior (lambda x draft theta)"), ("bo1_adj", "bo1 Sealed adjustment"), ("bo3_adj", "bo3 Sealed adjustment"), ("resid", "reconstruction residual")):
                if nm in o["piece"]:
                    L.append(f"| {lbl} | {_fmt_share(o['piece'][nm])} |")
            L.append(f"| all Sealed adjustments | {_fmt_share(o['sealed'])} |")
            L += ["", "| family | draft prior | Sealed adjustments |", "|---|---:|---:|"]
            for fam_n in FAMILIES:
                L.append(f"| {fam_n} | {_fmt_share(o['family'][fam_n].get('draft'))} | {_fmt_share(o['family'][fam_n]['sealed'])} |")
    L += ["", "RMS of (b_A - b_B) x sd (logit per SD of the feature) over (set x feature) cells, all refit sets:", "",
          "| format | family | total | draft part | Sealed part |", "|---|---|---:|---:|---:|"]
    for fm, d in st["all_refit_sets"].get("rms_dcoef_std", {}).items():
        for fam_n, v in d.items():
            L.append(f"| {fm} | {fam_n} | {v.get('total', float('nan')):.4f} | {v.get('draft', float('nan')):.4f} | {v.get('sealed', float('nan')):.4f} |")
    # ---- MX
    mx = res["mx"]
    L += ["", "## 4. The cross-set structure prior (MX)", "",
          "Between-set SD tau of the sets' structure effects (raw units; from the half models of the sets with a refit, mean over formats as crossset.other_set_effects builds it, converted with "
          "each half's sd) against the first-stage (draft; OM1: bo1) SE of the same coefficient. tau (noise-free) = sqrt(max(0, tau^2 - mean SE^2)) with the SE of the final estimates. "
          "Prior weight = (Laplace SE / prior SD)^2: 'default' = chain without MX (SD = sigma_theta); 'MX as built' = SD max(sigma_theta, tau), the floor in cardvalue.run_chain; "
          "'ideal' = SD tau. Medians over (set, half, feature):", "",
          "| group (non-gauge features unless stated) | n | median SE | median tau | median tau (noise-free) | weight default | weight MX as built | weight ideal (SD = tau) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for gname, v in mx["by_family"].items():
        if v.get("n"):
            L.append(f"| {gname} | {v['n']} | {v['se_data']:.4f} | {v['tau_std']:.4f} | {v['tau_free_std']:.4f} | {v['w_default']:.4f} | {v['w_as_built']:.4f} | {v['w_ideal']:.3f} |")
    L += [""]
    return "\n".join(L) + "\n"


# =====================================================================================================================
# driver
# =====================================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("terms")
    p.add_argument("sh")
    p.add_argument("out")
    p.add_argument("--sets", nargs="+", default=SETS)
    p = sub.add_parser("stage")
    p.add_argument("sh")
    p.add_argument("out")
    p.add_argument("--sets", nargs="+", default=STAGE_SETS)
    p = sub.add_parser("report")
    p.add_argument("sh")
    p.add_argument("out")
    p.add_argument("--sets", nargs="+", default=SETS)
    p.add_argument("--stage-sets", nargs="+", default=STAGE_SETS)
    p = sub.add_parser("structure")
    p.add_argument("sh")
    p.add_argument("decomp", help="the folder of the `terms` outputs (terms/SET.pkl)")
    p.add_argument("out")
    p.add_argument("--sets", nargs="+", default=SETS)
    p.add_argument("--stage-sets", nargs="+", default=STAGE_SETS)
    p.add_argument("--steps", nargs="+", default=["pools", "refit", "report"], choices=["pools", "refit", "report"])
    p = sub.add_parser("time-one")
    p.add_argument("set")
    p.add_argument("half", choices=("A", "B"))
    p.add_argument("sh")
    a = ap.parse_args(argv)
    if a.cmd == "terms":
        run_terms(Path(a.sh).resolve(), Path(a.out).resolve(), a.sets)
    elif a.cmd == "stage":
        for sc in a.sets:
            for h in ("A", "B"):
                stage_for_half(Path(a.sh).resolve(), sc, h, Path(a.out).resolve())
    elif a.cmd == "report":
        print(run_report(Path(a.sh).resolve(), Path(a.out).resolve(), a.sets, a.stage_sets))
    elif a.cmd == "structure":
        sh, decomp, out = Path(a.sh).resolve(), Path(a.decomp).resolve(), Path(a.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if "pools" in a.steps:
            run_structure_pools(sh, decomp, out, a.sets)
        if "refit" in a.steps:
            for sc in a.sets:
                for h in ("A", "B"):
                    structure_refit_half(sh, sc, h, out)
        if "report" in a.steps:
            print(run_structure_report(sh, out, a.sets, a.stage_sets))
    elif a.cmd == "time-one":
        sh = Path(a.sh).resolve()
        ev = split_events_of(sh, a.set)
        t0 = time.time()
        r = fit_draft_stage(a.set, home_of(sh, a.set, a.half), ev[a.half])
        print(f"draft-only: setup+fit {time.time() - t0:.0f}s, sigma {r['sigma']:.4f}, {r['n_games']} games, {r['n_events']} events, "
              f"converged {r['converged']}, max |beta| {np.nanmax(np.abs(r['beta'])):.3f}, median se {np.nanmedian(r['se']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
