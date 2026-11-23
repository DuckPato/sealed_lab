"""scripts/split_half_decompose.py: the pure arithmetic of the disagreement decomposition (synthetic numbers, no model fit)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("split_half_decompose", _SCRIPTS / "split_half_decompose.py")
sd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sd)


def _synthetic(seed=0, n=40):
    rng = np.random.default_rng(seed)
    bA, bB = rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)
    # castability-weighted copies: zero when the card is not in the deck, model-specific weights otherwise
    ina, inb = rng.random(n) < 0.5, rng.random(n) < 0.5
    cnt_a, cnt_b = ina * rng.integers(1, 3, n), inb * rng.integers(1, 3, n)
    wA_a, wB_a = cnt_a * rng.uniform(0.7, 1, n), cnt_a * rng.uniform(0.7, 1, n)
    wA_b, wB_b = cnt_b * rng.uniform(0.7, 1, n), cnt_b * rng.uniform(0.7, 1, n)
    return bA, bB, wA_a, wA_b, wB_a, wB_b, cnt_a, cnt_b


def test_card_contrib_equals_delta_difference_and_splits_exactly():
    bA, bB, wA_a, wA_b, wB_a, wB_b, *_ = _synthetic()
    tot, bp, wp = sd.card_T_contrib(bA, bB, wA_a, wA_b, wB_a, wB_b)
    # T_cards = Delta(D_A) - Delta(D_B), Delta(D) = L_A(D) - L_B(D), cards term of a model = beta . weights
    delta_a = bA @ wA_a - bB @ wB_a
    delta_b = bA @ wA_b - bB @ wB_b
    assert tot.sum() == pytest.approx(delta_a - delta_b, abs=1e-12)
    assert (bp + wp) == pytest.approx(tot, abs=1e-14)             # per card, exactly
    # identical models and weights: no disagreement at all
    z, zb, zw = sd.card_T_contrib(bA, bA, wA_a, wA_b, wA_a, wA_b)
    assert np.allclose(z, 0) and np.allclose(zb, 0) and np.allclose(zw, 0)
    # same weights, different betas: all of it is the beta part
    t2, b2, w2 = sd.card_T_contrib(bA, bB, wA_a, wA_b, wA_a, wA_b)
    assert np.allclose(w2, 0) and np.allclose(t2, b2)


def test_group_sums_conserve_the_total_and_reject_unknown_labels():
    bA, bB, wA_a, wA_b, wB_a, wB_b, ca, cb = _synthetic(1)
    tot, _, _ = sd.card_T_contrib(bA, bB, wA_a, wA_b, wB_a, wB_b)
    lab = np.array(["C", "U", "R", "M", "land", "basic"] * 7)[:len(tot)].astype(object)
    g = sd.group_sums(tot, lab, sd.RARITY_GROUPS)
    assert sum(g.values()) == pytest.approx(tot.sum(), abs=1e-12)
    pres = sd.presence_classes(ca, cb)
    assert sum(sd.group_sums(tot, pres, sd.PRES_GROUPS).values()) == pytest.approx(tot.sum(), abs=1e-12)
    with pytest.raises(ValueError):
        sd.group_sums(tot, np.array(["zzz"] * len(tot), dtype=object), sd.RARITY_GROUPS)


def test_term_T_is_additive_over_terms():
    rng = np.random.default_rng(2)
    terms = {t: rng.normal(size=4) for t in sd.TERMS}          # (A on D_A, A on D_B, B on D_A, B on D_B)
    t_sum = sum(sd.term_T(*v) for v in terms.values())
    tot = [sum(v[i] for v in terms.values()) for i in range(4)]           # the logits L_X(D)
    assert t_sum == pytest.approx(sd.term_T(*tot), abs=1e-12)
    # T = what the two models together gain from their own deck
    assert sd.term_T(*tot) == pytest.approx((tot[0] - tot[1]) + (tot[3] - tot[2]), abs=1e-12)


def test_slope_and_pp_conversion():
    assert sd.slope([0.0, 0.0]) == pytest.approx(0.25)
    assert sd.slope([np.log(3), -np.log(3)]) == pytest.approx(0.1875)          # p = 0.75 / 0.25 -> p(1-p) = 0.1875


def test_exposure_tertiles_are_balanced_and_basics_excluded():
    expo = np.array([5.0, 1, 9, 3, 7, 2, 8, 4, 6, 0, 100])
    elig = np.array([True] * 9 + [True, False])
    lab = sd.exposure_groups(expo, elig)
    assert lab[10] == "basic"
    counts = {g: int((lab == g).sum()) for g in ("T1", "T2", "T3")}
    assert sorted(counts.values()) == [3, 3, 4]
    assert set(lab[np.argsort(expo)[:3]]) == {"T1"}                # the three least exposed cards are the lowest tertile
    assert lab[np.argmax(np.where(elig, expo, -1))] == "T3"


def test_chain_parts_add_up_to_the_final_betas():
    rng = np.random.default_rng(3)
    D = rng.normal(0, 0.05, 30)
    lam = 0.7
    b1 = lam * D + rng.normal(0, 0.003, 30)               # stacked bo1 estimate: prior + adjustment
    b3 = b1 + rng.normal(0, 0.002, 30)
    p = sd.chain_parts(D, lam, b1, b3)
    assert sum(p["bo1"].values()) == pytest.approx(b1)
    assert sum(p["bo3"].values()) == pytest.approx(b3)
    assert p["bo3"]["draft"] == pytest.approx(lam * D)
    assert p["bo1"]["bo3_adj"] == pytest.approx(0)
    # stacking weight 0: the final estimate is lam * D, no Sealed adjustment
    q = sd.chain_parts(D, lam, lam * D, lam * D)
    assert np.allclose(q["bo1"]["bo1_adj"], 0) and np.allclose(q["bo3"]["bo3_adj"], 0)
    # bo1 is no prior: bo3's prior is lam3 * D
    r = sd.chain_parts(D, None, None, 0.5 * D + 0.01, bo1_is_prior=False, lam3=0.5)
    assert sum(r["bo3"].values()) == pytest.approx(0.5 * D + 0.01)


def test_lambda_split_is_exact_and_stage_pieces_give_the_same_T():
    bA, bB, wA_a, wA_b, wB_a, wB_b, *_ = _synthetic(4)
    rng = np.random.default_rng(5)
    DA, DB = rng.normal(0, 0.05, len(bA)), rng.normal(0, 0.05, len(bA))
    la, lb = 0.8, 0.65
    (cA, cB), (sA, sB) = sd.lambda_split(DA, DB, la, lb)
    assert cA + sA == pytest.approx(la * DA) and cB + sB == pytest.approx(lb * DB)
    whole = sd.card_T_contrib(la * DA, lb * DB, wA_a, wA_b, wB_a, wB_b)[0].sum()
    parts = sd.card_T_contrib(cA, cB, wA_a, wA_b, wB_a, wB_b)[0].sum() + sd.card_T_contrib(sA, sB, wA_a, wA_b, wB_a, wB_b)[0].sum()
    assert whole == pytest.approx(parts, abs=1e-12)                # linear in beta: any exact split of the betas splits T


def test_weighted_split_moments_add_up():
    rng = np.random.default_rng(6)
    v = rng.uniform(0, 5, 50)
    dd, ds = rng.normal(0, 0.03, 50), rng.normal(0, 0.003, 50)
    m = sd.weighted_split_moments(v, {"draft": dd, "sealed": ds})
    assert m["share_draft"] + m["share_sealed"] + m["share_cov"] == pytest.approx(1.0, abs=1e-12)
    assert m["rms"] == pytest.approx(np.sqrt((v * (dd + ds) ** 2).sum() / v.sum()))
    z = sd.weighted_split_moments(np.zeros(5), {"draft": np.ones(5), "sealed": np.ones(5)})
    assert z["sum_w"] == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# structure / mana feature split (the `structure` subcommand)
# ---------------------------------------------------------------------------------------------------------------------

def _synthetic_decks(rng, n_decks=2):
    """phi rows (FEATURES order, 20 columns) of 40-card decks: land / creature / non-creature / curve counts add up as in structure_counts."""
    F = sd.FEATURES
    rows = []
    for _ in range(n_decks):
        nl = int(rng.integers(15, 19))
        ns = 40 - nl
        cre = int(rng.integers(8, ns - 4))
        curve = rng.multinomial(ns, [0.1, 0.25, 0.25, 0.2, 0.12, 0.08]).astype(float)
        phi = rng.normal(size=len(F))
        for name, v in zip(["n_lands", "n_lands_dev2", "n_creatures", "n_noncreature"], [nl, (nl - 17) ** 2, cre, ns - cre]):
            phi[F.index(name)] = v
        for b, name in enumerate(["curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6"]):
            phi[F.index(name)] = curve[b]
        rows.append(phi)
    return rows


def test_families_partition_the_features_and_alignment_roundtrips():
    all_f = [f for fs in sd.FAMILIES.values() for f in fs]
    assert sorted(all_f) == sorted(sd.FEATURES) and len(all_f) == len(set(all_f))
    assert sd.NF == 20 and set(sd.GAUGE_BLOCK) <= set(sd.FEATURES)
    cov = sd.coarse_index()
    assert sorted(np.concatenate(list(cov.values())).tolist()) == list(range(sd.NF))
    names = [f for f in sd.FEATURES if f != "dead_cards"]            # a model without castability
    v, present = sd.align_features(names, np.arange(len(names), dtype=float) + 1.0)
    assert not present[sd.FEATURES.index("dead_cards")] and present.sum() == 19 and v[sd.FEATURES.index("dead_cards")] == 0.0
    assert v[sd.FEATURES.index("n_lands")] == 1.0


def test_per_feature_split_sums_to_the_structure_part_of_T_and_mu_terms_cancel():
    rng = np.random.default_rng(10)
    n = sd.NF
    thA, thB = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)                        # standardised coefficients of the two models
    muA, muB = rng.normal(10, 3, n), rng.normal(10, 3, n)                          # DIFFERENT standardisations: mu and sd per model
    sdA, sdB = rng.uniform(0.5, 3, n), rng.uniform(0.5, 3, n)
    da, db = _synthetic_decks(rng)
    dead = sd.FEATURES.index("dead_cards")
    phi = {("A", "A"): da.copy(), ("A", "B"): db.copy(), ("B", "A"): da.copy(), ("B", "B"): db.copy()}
    phi[("B", "A")][dead] += 0.3                                                   # the dead-cards column differs between the models
    phi[("B", "B")][dead] -= 0.2

    def S(th, mu, sd_, p):                                                          # the model's struct+mana logit: theta . ((phi - mu) / sd)
        return float(th @ ((p - mu) / sd_))

    sm = {k: S(thA if k[0] == "A" else thB, muA if k[0] == "A" else muB, sdA if k[0] == "A" else sdB, p) for k, p in phi.items()}
    T = sd.term_T(sm[("A", "A")], sm[("A", "B")], sm[("B", "A")], sm[("B", "B")])
    bA, bB = thA / sdA, thB / sdB                                                   # raw units: the mu terms are gone
    tot, bpart, ppart = sd.feature_T_parts(bA, bB, phi[("A", "A")], phi[("A", "B")], phi[("B", "A")], phi[("B", "B")])
    assert tot.sum() == pytest.approx(T, abs=1e-12)
    assert bpart + ppart == pytest.approx(tot, abs=1e-14)
    assert np.allclose(np.delete(ppart, dead), 0.0)                                 # only the dead-cards column has a phi disagreement
    # identical models and features: nothing
    z, zb, zp = sd.feature_T_parts(bA, bA, phi[("A", "A")], phi[("A", "B")], phi[("A", "A")], phi[("A", "B")])
    assert np.allclose(z, 0) and np.allclose(zb, 0) and np.allclose(zp, 0)
    # the family sums add up to the total
    assert sum(tot[ix].sum() for ix in sd.family_index().values()) == pytest.approx(T, abs=1e-12)


def test_gauge_null_directions_leave_T_unchanged_and_the_gauge_free_distance_ignores_them():
    rng = np.random.default_rng(11)
    da, db = _synthetic_decks(rng)
    idx = np.arange(sd.NF)
    U = sd.gauge_null_vectors(idx)
    assert U.shape == (sd.NF, 2)
    assert np.allclose(U.T @ (da - db), 0)                                          # equal-size decks: the null directions do not move any difference
    bA, bB = rng.normal(0, 0.05, sd.NF), rng.normal(0, 0.05, sd.NF)
    t0 = sd.feature_T_parts(bA, bB, da, db, da, db)[0].sum()
    t1 = sd.feature_T_parts(bA + U @ np.array([0.7, -0.4]), bB, da, db, da, db)[0].sum()
    assert t0 == pytest.approx(t1, abs=1e-12)
    # the Mahalanobis distance is unchanged by a shift of d along a null direction, even with a covariance that has no variance there
    L = rng.normal(size=(sd.NF, sd.NF))
    cov = L @ L.T + np.eye(sd.NF) * 0.1
    P = np.eye(sd.NF) - U @ np.linalg.pinv(U)                                       # project the null directions out: zero variance there (a sandwich does that)
    cov = P @ cov @ P
    scale = np.ones(sd.NF)
    d = rng.normal(size=sd.NF)
    m0, df0 = sd.identifiable_mahalanobis(d, cov, idx, scale)
    m1, df1 = sd.identifiable_mahalanobis(d + U @ np.array([5.0, -3.0]), cov, idx, scale)
    assert df0 == df1 == sd.NF - 2
    assert m0 == pytest.approx(m1, rel=1e-9)
    # a subset without the whole gauge block has no null direction
    rest = [j for j, f in enumerate(sd.FEATURES) if f not in sd.GAUGE_BLOCK]
    assert sd.gauge_null_vectors(rest).shape[1] == 0
    assert sd.identifiable_mahalanobis(d, cov, rest, scale)[1] == len(rest)


def test_propagate_cov_diagonal_matches_the_chains_propagate_se_formula():
    rng = np.random.default_rng(12)
    k = 6
    A = rng.normal(size=(k, k))
    data, prev = A @ A.T * 0.01, (A.T @ A) * 0.02
    wp = rng.uniform(0.05, 0.9, k)
    lam = 0.7
    # w = 1: se^2 = se_cond^2 + w_prior^2 (lam se_prev)^2          w < 1: se^2 = (w se_cond)^2 + (1 - w (1 - w_prior))^2 (lam se_prev)^2   (cardvalue.propagate_se)
    c1 = sd.propagate_cov(data, wp, lam, 1.0, prev)
    assert np.diag(c1) == pytest.approx(np.diag(data) + wp ** 2 * lam ** 2 * np.diag(prev), rel=1e-12)
    c2 = sd.propagate_cov(data, wp, lam, 0.4, prev)
    assert np.diag(c2) == pytest.approx((0.4 ** 2) * np.diag(data) + (1 - 0.4 * (1 - wp)) ** 2 * lam ** 2 * np.diag(prev), rel=1e-12)
    assert np.allclose(c2, c2.T) and np.linalg.eigvalsh(c2).min() > -1e-12
    # stacking weight 0: the estimate IS the (scaled) prior mean, so its covariance is lam^2 * prev
    c0 = sd.propagate_cov(data, np.zeros(k), lam, 0.0, prev)
    assert c0 == pytest.approx(lam ** 2 * prev, rel=1e-12)


def test_chi2_cells_and_raw_cov():
    r = sd.chi2_cells([1.0, -2.0, 0.0], [1.0, 4.0, 9.0])
    assert r["chi2"] == pytest.approx(1.0 + 1.0) and r["df"] == 3
    c = sd.raw_cov(np.array([[4.0, 2.0], [2.0, 9.0]]), np.array([2.0, 3.0]))
    assert c == pytest.approx(np.array([[1.0, 1 / 3], [1 / 3, 1.0]]))
    m, df = sd.mahalanobis([1.0, 1.0], np.eye(2) * 2)
    assert m == pytest.approx(1.0) and df == 2
