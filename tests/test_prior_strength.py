"""Roadmap 2.1 / 2.3 / 2.1b: time-split prior strength + stacking, the confirmation split, per-format recency half-lives."""
from __future__ import annotations

import types

import numpy as np
import pytest
import scipy.sparse as sp

from sealed_lab.model import cardvalue, design, glm, pipeline, validate


# ------------------------------------------------------------------------------------------------ 2.1: time folds + stacking

def _d(base, name):
    (base / name).mkdir(exist_ok=True)
    return str(base / name)


def _stage(n_groups=120, per=8, p=30, seed=0, effect=0.15, prior=None, name="bo1"):
    """A synthetic Sealed stage: one deck per game, events in start-time order, true card effects `effect` * N(0, 1)."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    group = np.repeat(np.arange(n_groups), per).astype(np.int32)
    X = sp.random(n, p, density=0.25, random_state=seed + 4, format="csr")
    X.data[:] = 1.0
    beta = rng.normal(size=p) * effect
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.1 + X @ beta)))).astype(float)
    C = sp.csr_matrix(np.ones((n, 1)))
    prob = glm.Problem(y=y, C=C, D=X.tocsr(), idx=np.arange(n, dtype=np.int32))
    # shuffled event ids so that the group index is NOT the time order
    start = np.datetime64("2026-01-01", "us") + (np.random.default_rng(seed + 9).permutation(n_groups) * 86_400_000_000).astype("timedelta64[us]")
    stage = types.SimpleNamespace(name=name, n_groups=n_groups, n_games=n, prob=prob, group=group, group_start=start,
                                  enc=types.SimpleNamespace(sigma=np.array([10.0])), w=None)
    return stage, beta


def test_time_folds_are_consecutive_blocks_in_time_order():
    stage, _ = _stage()
    fold = cardvalue.time_folds(stage)
    assert set(np.unique(fold)) == set(range(cardvalue.N_TIME_FOLDS))
    start = stage.group_start.astype("int64")
    for f in range(cardvalue.N_TIME_FOLDS - 1):
        assert start[fold == f].max() < start[fold == f + 1].min()              # blocks never interleave
    games = np.bincount(fold[stage.group], minlength=cardvalue.N_TIME_FOLDS)
    assert games.max() - games.min() <= 2 * 8                                    # about equal numbers of games


def test_forward_folds_never_train_on_later_events(tmp_path):
    """Flipping every label of the events AFTER block f must not change block f's held-out loss: the model never saw them."""
    stage, _ = _stage(seed=3)
    fold_g = cardvalue.time_folds(stage)[stage.group]
    p = stage.prob.p
    spec = cardvalue.make_spec(stage, p, 0, "bo1")
    pm = np.zeros(stage.prob.k + p)

    def run(y, f, path):
        pr = glm.Problem(y=y, C=stage.prob.C, D=stage.prob.D, idx=stage.prob.idx)
        glm.save_problem(path, pr, fold=fold_g.astype(np.int8), w=np.ones(len(y)), group=stage.group)
        glm._PROB_CACHE.clear()
        return glm.cv_fold_task(dict(path=path, fold=f, spec=spec, prior_mean=pm, grid=[0.05, 0.2], x0=None, maxiter=200,
                                     scheme="forward", fold_game=fold_g))

    for f in (1, 3):
        base = run(stage.prob.y, f, str(tmp_path / f"a{f}.npz"))
        y2 = stage.prob.y.copy()
        y2[fold_g > f] = 1.0 - y2[fold_g > f]
        flipped = run(y2, f, str(tmp_path / f"b{f}.npz"))
        np.testing.assert_allclose(base["loss"], flipped["loss"], rtol=0, atol=1e-9)
        assert base["n_va"] == int((fold_g == f).sum())
        # while flipping an EARLIER event does change it (the training data are used)
        y3 = stage.prob.y.copy()
        y3[fold_g < f] = 1.0 - y3[fold_g < f]
        assert not np.allclose(base["loss"], run(y3, f, str(tmp_path / f"c{f}.npz"))["loss"], atol=1e-6)


def test_fold_priors_never_see_later_events(tmp_path):
    """Leak-free time folds: flipping every label of the events that start at or after block f (in the PRIOR stage and in the
    target stage) leaves the prior P of block f, and so the block's held-out loss, unchanged; earlier labels do change it."""
    tgt, _ = _stage(seed=21, n_groups=150, per=8, p=30, name="bo1")
    pri, beta = _stage(seed=22, n_groups=150, per=8, p=30, name="draft")
    # the Sealed games follow the draft stage's card effects, so lambda > 0 and the prior carries real information
    tgt.prob.y[:] = (np.random.default_rng(5).random(tgt.prob.n) < 1 / (1 + np.exp(-(0.1 + tgt.prob.D @ beta)))).astype(float)
    stages = {"draft": pri, "bo1": tgt}
    fits = cardvalue.run_chain(stages, 30, 0, _d(tmp_path, "fit"), 2, seed=1, opts=cardvalue.ChainOpts("time", True, leak_free=False))
    fits["draft"].sigma = 0.3                                       # a loose draft stage, so that its estimates are not shrunk to ~0
    base = cardvalue.fold_priors(stages, fits, ["draft"], "bo1", _d(tmp_path, "p0"), 2, seed=1)
    assert set(base) == set(range(1, cardvalue.N_TIME_FOLDS)) and all(v.shape == (30,) for v in base.values())
    fold_g = cardvalue.time_folds(tgt)
    f = 3
    t_f = tgt.group_start[fold_g == f].min()

    def flip(stage, mask):
        y = stage.prob.y.copy()
        y[mask] = 1.0 - y[mask]
        ns = types.SimpleNamespace(**vars(stage))
        ns.prob = glm.Problem(y=y, C=stage.prob.C, D=stage.prob.D, idx=stage.prob.idx)
        return ns

    later = {"draft": pri.group_start[pri.group] >= t_f, "bo1": fold_g[tgt.group] >= f}
    flipped = {n: flip(st, later[n]) for n, st in stages.items()}
    other = cardvalue.fold_priors(flipped, fits, ["draft"], "bo1", _d(tmp_path, "p1"), 2, seed=1)
    np.testing.assert_allclose(other[f], base[f], rtol=0, atol=1e-9)
    for g in range(1, f):                                           # earlier blocks see even less
        np.testing.assert_allclose(other[g], base[g], rtol=0, atol=1e-9)
    assert not np.allclose(other[cardvalue.N_TIME_FOLDS - 1], base[cardvalue.N_TIME_FOLDS - 1], atol=1e-6)    # later priors do use them
    # the block's own loss, with M fitted on top of that prior, is unchanged too (its labels are the ones being scored)
    spec = fits["bo1"].spec

    def block_loss(st, prior, path):
        fog = fold_g[tgt.group]
        glm.save_problem(path, st.prob, fold=fog.astype(np.int8), w=np.ones(st.prob.n), group=tgt.group)
        glm._PROB_CACHE.clear()
        return glm.cv_fold_task(dict(path=path, fold=f, spec=spec, prior_mean=cardvalue.prior_vector(st.prob.k, prior), grid=[0.05],
                                     x0=None, maxiter=200, scheme="forward", fold_game=fog))["loss"]
    np.testing.assert_allclose(block_loss(tgt, base[f], str(tmp_path / "l0.npz")),
                               block_loss(flip(tgt, fold_g[tgt.group] > f), other[f], str(tmp_path / "l1.npz")), rtol=0, atol=1e-9)
    # earlier events do change the prior (the training data are used)
    early = {"draft": pri.group_start[pri.group] < t_f, "bo1": fold_g[tgt.group] < f}
    changed = cardvalue.fold_priors({n: flip(st, early[n]) for n, st in stages.items()}, fits, ["draft"], "bo1", _d(tmp_path, "p2"),
                                    2, seed=1)
    assert not np.allclose(changed[f], base[f], atol=1e-6)


def test_stack_w0_reproduces_the_prior_and_w1_the_plain_fit(tmp_path):
    stage, _ = _stage(seed=5)
    p = stage.prob.p
    rng = np.random.default_rng(1)
    prior = rng.normal(size=p) * 0.1
    spec = cardvalue.make_spec(stage, p, 0, "bo1")
    coef, _se, _res = cardvalue.fit_stage(stage, prior, spec, 0.2, with_se=False)
    k = stage.prob.k
    s0 = glm.stack_coef(stage.prob, coef, prior, 0.0, spec.sig_cov)
    np.testing.assert_array_equal(s0[k:], prior)                                  # deck part = the prior mean exactly
    # the prediction is the prior-only model P: its deck score, covariates refitted given it (offset) - the report's baseline P
    off = (stage.prob.D @ prior)[stage.prob.idx]
    cov_p = glm.fit(glm.Problem(y=stage.prob.y, C=stage.prob.C, offset=off), np.zeros(k), spec.sig_cov, maxiter=400).coef
    np.testing.assert_allclose(stage.prob.eta(s0), stage.prob.C @ cov_p + off, atol=1e-6)
    np.testing.assert_array_equal(glm.stack_coef(stage.prob, coef, prior, 1.0, spec.sig_cov), coef)
    mid = glm.stack_coef(stage.prob, coef, prior, 0.4, spec.sig_cov)
    np.testing.assert_allclose(mid[k:], 0.6 * prior + 0.4 * coef[k:])


def test_bootstrap_worker_stacks_each_replicate_towards_its_own_prior(tmp_path):
    stage, _ = _stage(seed=6, n_groups=60)
    p = stage.prob.p
    spec = cardvalue.make_spec(stage, p, 0, "bo1")
    path = cardvalue.stage_files(stage, str(tmp_path), 0)
    pm = cardvalue.prior_vector(stage.prob.k, np.random.default_rng(2).normal(size=p) * 0.1)

    def one(w):
        return glm.boot_chain_task(dict(rep=1, stages=[dict(path=path, spec=spec, s=0.2, x0=None, prior_mean=pm, deck_from_prev=False,
                                                         seed=3, maxiter=200, stack_w=w)]))["coefs"][0]

    full, zero, half = one(None), one(0.0), one(0.5)
    k = stage.prob.k
    np.testing.assert_allclose(zero[k:], pm[k:], atol=1e-12)
    np.testing.assert_allclose(half[k:], pm[k:] + 0.5 * (full[k:] - pm[k:]), atol=1e-9)


def test_time_cv_extends_the_grid_and_chooses_w_by_held_out_loss(tmp_path):
    n_jobs = 2
    # (a) the Sealed games carry real card effects the prior knows nothing about: the plain fit M must (nearly) win (w close to 1)
    stage, beta = _stage(seed=11, n_groups=300, per=10, p=40, effect=0.3)
    s, info = cardvalue.select_sigma(stage, np.zeros(stage.prob.p), stage.prob.p, 0, _d(tmp_path, "a"), n_jobs, seed=1,
                                     opts=cardvalue.ChainOpts("time", True))
    assert info["scheme"] == "forward" and len(info["grid"]) == 10 and info["grid"][0] < 0.002
    assert info["w"] >= 0.7 and len(info["w_loss"]) == len(cardvalue.STACK_GRID)
    # (b) the stacking weights are scored at the chosen sigma on the same folds: here the prior IS the truth and a loose sigma
    #     overfits, so the prior-only model P (w = 0) must score best, and w = 1 must equal the plain fold loss
    stage, beta = _stage(seed=12, n_groups=300, per=10, p=40, effect=0.3)
    fold_g = cardvalue.time_folds(stage)[stage.group]
    path = _d(tmp_path, "b") + "/prob.npz"
    glm.save_problem(path, stage.prob, fold=fold_g.astype(np.int8), w=np.ones(stage.prob.n), group=stage.group)
    spec = cardvalue.make_spec(stage, stage.prob.p, 0, "bo1")
    pm = cardvalue.prior_vector(stage.prob.k, beta)
    task = dict(path=path, fold=3, spec=spec, prior_mean=pm, x0=None, maxiter=300, scheme="forward", fold_game=fold_g)
    plain = glm.cv_fold_task(dict(task, grid=[0.3]))
    r = glm.stack_fold_task(dict(task, sigma=0.3, w_grid=[0.0, 0.5, 1.0]))
    lw = r["loss_w"]
    assert lw[2] == pytest.approx(plain["loss"][0], rel=1e-6) and lw[0] < lw[1] < lw[2]
    assert r["n_va"] == plain["n_va"] == int((fold_g == 3).sum())
    # random folds / no prior stage: the original rule (8 grid points, no stacking weight)
    s2, info2 = cardvalue.select_sigma(stage, beta, stage.prob.p, 0, _d(tmp_path, "c"), n_jobs, seed=1, opts=cardvalue.LEGACY_OPTS)
    assert info2["scheme"] == "random" and len(info2["grid"]) == 8 and "w" not in info2
    s3, info3 = cardvalue.select_sigma(stage, np.zeros(stage.prob.p), stage.prob.p, 0, _d(tmp_path, "d"), n_jobs, seed=1,
                                       has_prior=False)
    assert "w" not in info3


def test_stacked_stage_fit_is_the_next_stage_prior_and_se_interpolates(tmp_path):
    stage, beta = _stage(seed=13, n_groups=200, per=10, p=40, effect=0.3)
    stages = {"bo1": stage}
    # a draft stage that knows the truth: bo1 stacks towards it
    dstage, _ = _stage(seed=14, n_groups=200, per=10, p=40, effect=0.3, name="draft")
    dstage.prob.y[:] = (np.random.default_rng(15).random(dstage.prob.n) < 1 / (1 + np.exp(-(0.1 + dstage.prob.D @ beta)))).astype(float)
    fits = cardvalue.run_chain({"draft": dstage, "bo1": stage}, 40, 0, str(tmp_path), 2, seed=1, opts=cardvalue.ChainOpts("time", True))
    f = fits["bo1"]
    assert f.cv_scheme == "forward" and 0.0 <= f.w <= 1.0 and f.w_loss
    if f.w < 1.0:
        np.testing.assert_allclose(f.coef[f.k:], f.prior_deck + f.w * (f.coef_m[f.k:] - f.prior_deck), atol=1e-12)
    assert np.all(np.isfinite(f.se_total)) and np.all(f.se_total > 0)
    legacy = cardvalue.run_chain({"draft": dstage, "bo1": stage}, 40, 0, _d(tmp_path, "l"), 2, seed=1, opts=cardvalue.LEGACY_OPTS)["bo1"]
    assert legacy.w == 1.0 and legacy.coef_m is None and legacy.cv_scheme == "random" and len(legacy.cv_grid) == 8


# ------------------------------------------------------------------------------------------------ 2.3: confirmation split

def _halves(a, b):
    """halves dict from (gain, lo, hi) for A and (gain, lo, hi, lo90) for B."""
    return dict(A=dict(gain=a[0], lo=a[1], hi=a[2], lo90=a[1], hi90=a[2]), B=dict(gain=b[0], lo=b[1], hi=b[2], lo90=b[3], hi90=b[2]))


def _val(gain, lo, hi, halves, fmt="bo1"):
    return {fmt: dict(status="ok", comp=dict(gain=gain, lo=lo, hi=hi, halves=halves))}


def test_component_failing_the_second_half_is_not_applied():
    ok_b = (0.002, 0.0, 0.004, 0.0008)
    val = _val(0.003, 0.001, 0.005, _halves((0.0035, 0.0006, 0.0064), ok_b))
    d = pipeline.decide_component("auto", "comp", val, True, "", "the thing")
    assert d["applied"] and d["confirmation"]["confirmed"] == ["bo1"]
    # half B: same sign but the one-sided 90% bound is below zero -> not confirmed
    val = _val(0.003, 0.001, 0.005, _halves((0.0035, 0.0006, 0.0064), (0.002, -0.002, 0.006, -0.0004)))
    d = pipeline.decide_component("auto", "comp", val, True, "", "the thing")
    assert not d["applied"] and "NOT confirmed" in d["reason"] and d["confirmation"]["old_rule_applied"]
    # half B has the opposite sign
    val = _val(0.003, 0.001, 0.005, _halves((0.0035, 0.0006, 0.0064), (-0.001, -0.004, 0.002, -0.003)))
    assert not pipeline.decide_component("auto", "comp", val, True, "", "the thing")["applied"]
    # half A alone does not pass although the whole hold-out does
    val = _val(0.003, 0.001, 0.005, _halves((0.0035, -0.0004, 0.0064), ok_b))
    d = pipeline.decide_component("auto", "comp", val, True, "", "the thing")
    assert not d["applied"] and "earlier half" in d["reason"]
    # the confirmation can be switched off (the original rule); a missing half split fails closed (audit 2026-11-24, C1)
    val = _val(0.003, 0.001, 0.005, _halves((0.0035, -0.0004, 0.0064), (-0.001, -0.004, 0.002, -0.003)))
    assert pipeline.decide_component("auto", "comp", val, True, "", "the thing", confirm=False)["applied"]
    plain = {"bo1": dict(status="ok", comp=dict(gain=0.003, lo=0.001, hi=0.005))}
    d = pipeline.decide_component("auto", "comp", plain, True, "", "the thing")
    assert not d["applied"] and "half-split numbers are missing" in d["reason"]
    # a format that is materially and significantly worse on one half vetoes it, even if another format is confirmed
    both = {**_val(0.003, 0.001, 0.005, _halves((0.0035, 0.0006, 0.0064), ok_b), "bo3"),
            **_val(-0.0001, -0.0005, 0.0003, _halves((-0.0004, -0.0007, -0.0002), (0.0002, -0.0003, 0.0007, -0.0001)), "bo1")}
    d = pipeline.decide_component("auto", "comp", both, True, "", "the thing")
    assert not d["applied"] and "WORSE" in d["reason"]


def test_castability_is_checked_on_both_halves():
    def val(a, b):
        return {"bo1": dict(status="ok", castability=dict(gain=0.0, lo=-0.001, hi=0.001, halves=_halves(a, b)))}
    fine = pipeline.decide_castability("auto", val((0.0003, -0.001, 0.001), (-0.0001, -0.001, 0.001, -0.0008)), True)
    assert fine["applied"]
    bad_b = pipeline.decide_castability("auto", val((0.0003, -0.001, 0.001), (-0.0006, -0.0009, -0.0003, -0.0008)), True)
    assert not bad_b["applied"] and "WORSE" in bad_b["reason"]
    assert pipeline.decide_castability("auto", val((0.0003, -0.001, 0.001), (-0.0006, -0.0009, -0.0003, -0.0008)), True, confirm=False)["applied"]


def test_half_gains_split_events_by_time_and_resample_within_halves():
    rng = np.random.default_rng(0)
    G, per = 40, 50
    groups = np.repeat(np.arange(G), per)
    y = (rng.random(G * per) < 0.55).astype(float)
    p_base = np.full(G * per, 0.55)
    p_new = np.where(groups < 20, np.clip(0.55 + 0.1 * (y - 0.55), 0.01, 0.99), 0.55)       # better only on events 0..19
    is_first = np.arange(G) < 20
    h = validate.half_gains(y, p_base, p_new, groups, is_first, 400, 1)
    assert h["A"]["gain"] > 0 and h["A"]["lo"] > 0 and h["A"]["n_events"] == 20
    assert h["B"]["gain"] == pytest.approx(0.0, abs=1e-12) and h["B"]["n_events"] == 20 and h["B"]["lo90"] <= 0 <= h["B"]["hi90"]
    assert h["A"]["lo90"] >= h["A"]["lo"] and h["A"]["hi90"] <= h["A"]["hi"]


# ------------------------------------------------------------------------------------------------ 2.1b: per-format half-lives

def _inner(gain_by_h):
    rows = [dict(half_life=None, logloss=0.6, diff=0.0, lo=0.0, hi=0.0)]
    rows += [dict(half_life=h, logloss=0.6 - g, diff=g, lo=g - 0.01, hi=g + 0.01) for h, g in gain_by_h.items()]
    return dict(status="ok", rows=rows)


def test_half_lives_are_chosen_and_applied_per_format():
    inner = {"bo1": _inner({60.0: -0.001, 7.0: -0.004}), "bo3": _inner({60.0: 0.001, 7.0: 0.003})}
    chosen = {f: validate.choose_half_life({f: v}) for f, v in inner.items()}
    assert chosen == {"bo1": None, "bo3": 7.0}                                  # the pooled choice would sum the two formats
    assert validate.choose_half_life(inner) == 7.0 or validate.choose_half_life(inner) is None
    good = (0.0035, 0.0006, 0.0064)
    halves = _halves(good, (0.002, 0.0, 0.004, 0.0008))
    val = {"bo3": dict(status="ok", recency=dict(half_life=7.0, gain=0.003, lo=0.001, hi=0.005, halves=halves))}
    rec = pipeline.decide_recency_formats("auto", chosen, inner, val)
    assert rec["applied"] and rec["half_life"] == {"bo1": None, "bo3": 7.0} and rec["scope"] == "format"
    assert rec["formats"]["bo1"]["applied"] is False and rec["formats"]["bo3"]["applied"] is True
    # bo3 failing its confirmation leaves BOTH formats on equal weights, bo1 passing alone would not depend on bo3
    bad = {"bo3": dict(status="ok", recency=dict(half_life=7.0, gain=0.003, lo=0.001, hi=0.005,
                                                 halves=_halves(good, (0.002, -0.002, 0.006, -0.0004))))}
    rec2 = pipeline.decide_recency_formats("auto", chosen, inner, bad)
    assert not rec2["applied"] and rec2["half_life"] == {"bo1": None, "bo3": None}


def _stage_data(name, n_groups=4):
    start = np.datetime64("2026-06-01", "us") + (np.arange(n_groups) * 86_400_000_000).astype("timedelta64[us]")
    return design.StageData(name=name, games=None, enc=None, prob=None, group=np.arange(n_groups, dtype=np.int32), group_ids=[],
                            group_start=start, deck_rows=None, exposure=None, ref_row=None, n_games=n_groups, n_groups=n_groups)


def test_weight_stages_per_format_and_draft_always_equal():
    stages = {n: _stage_data(n) for n in ("draft", "bo1", "bo3")}
    t_ref = np.datetime64("2026-06-05")
    per = validate.weight_stages(stages, t_ref, {"bo3": 2.0})
    assert per["draft"].w is None and per["bo1"].w is None                         # not listed: equal weights
    assert per["bo3"].w is not None and per["bo3"].w[0] < per["bo3"].w[-1] <= 1.0
    both = validate.weight_stages(stages, t_ref, {"bo1": 2.0, "bo3": 30.0})
    assert both["bo1"].w is not None and both["bo3"].w is not None and both["draft"].w is None
    assert both["bo1"].w[0] < both["bo3"].w[0]                                     # each stage uses its own half-life
    pooled = validate.weight_stages(stages, t_ref, 2.0)                           # a plain number: every stage but the draft
    assert pooled["draft"].w is None and pooled["bo1"].w is not None and pooled["bo3"].w is not None
    np.testing.assert_allclose(per["bo3"].w, pooled["bo3"].w)
