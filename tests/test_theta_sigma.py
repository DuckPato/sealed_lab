"""theta_sigma: the structure terms' prior SD of the Sealed stages, by rule (clip(4 sigma), the default) or chosen by the card sigma's CV.

All fits here are tiny synthetic stages with n_jobs = 1 (no process pool)."""
from __future__ import annotations

import inspect
import types

import numpy as np
import pytest
import scipy.sparse as sp

from sealed_lab.model import cardvalue, glm, pipeline


def _stage(n_groups=160, per=8, n_card=20, n_feat=3, theta=(0.0, 0.0, 0.0), seed=0, name="bo1"):
    """A Sealed stage with n_card card columns then n_feat structure columns (dense, standardised-like), true structure effects `theta`."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    group = np.repeat(np.arange(n_groups), per).astype(np.int32)
    Xc = sp.random(n, n_card, density=0.25, random_state=seed + 4, format="csr")
    Xc.data[:] = 1.0
    Xs = sp.csr_matrix(rng.normal(size=(n, n_feat)))
    X = sp.hstack([Xc, Xs], format="csr")
    beta = np.concatenate([rng.normal(size=n_card) * 0.1, np.asarray(theta, dtype=float)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.1 + X @ beta)))).astype(float)
    prob = glm.Problem(y=y, C=sp.csr_matrix(np.ones((n, 1))), D=X.tocsr(), idx=np.arange(n, dtype=np.int32))
    start = np.datetime64("2026-01-01", "us") + (np.random.default_rng(seed + 9).permutation(n_groups) * 86_400_000_000).astype("timedelta64[us]")
    return types.SimpleNamespace(name=name, n_groups=n_groups, n_games=n, prob=prob, group=group, group_start=start,
                                 enc=types.SimpleNamespace(sigma=np.array([10.0])), w=None)


def test_chain_opts_validate_and_keep_the_rule_cache_key():
    assert cardvalue.ChainOpts().theta_sigma == "rule"
    with pytest.raises(ValueError):
        cardvalue.ChainOpts(theta_sigma="auto")
    # the rule leaves the key (used by the validation cache) exactly as before; cv adds the option
    assert cardvalue.ChainOpts().key() == dict(sigma_cv="time", stack=True, leak_free=False)
    assert cardvalue.ChainOpts(theta_sigma="cv").key() == dict(sigma_cv="time", stack=True, leak_free=False, theta_sigma="cv")
    o = cardvalue.ChainOpts(theta_sigma="cv")
    assert o.theta_cv("bo1") and o.theta_cv("bo3") and not o.theta_cv("draft")
    assert not cardvalue.ChainOpts().theta_cv("bo1")


def test_sigma_spec_vector_rule_cv_and_floor():
    sc = np.array([10.0])
    spec = glm.SigmaSpec(sig_cov=sc, n_card=2, n_theta=3, kappa=4.0, th_lo=0.05, th_hi=0.5)
    for s in (0.002, 0.02, 0.2):                                    # th_abs None: the old formula, bit for bit
        np.testing.assert_array_equal(spec.vector(s), np.concatenate([sc, np.full(2, s), np.full(3, np.clip(4.0 * s, 0.05, 0.5))]))
    fixed = glm.SigmaSpec(sig_cov=sc, n_card=2, n_theta=3, th_lo=0.05, th_hi=0.5, th_abs=0.0125)
    np.testing.assert_array_equal(fixed.vector(0.3)[3:], np.full(3, 0.0125))      # independent of the card sigma and of the clip
    floored = glm.SigmaSpec(sig_cov=sc, n_card=2, n_theta=3, th_abs=0.0125, th_floor=np.array([0.0, 0.03, 0.01]))
    np.testing.assert_array_equal(floored.vector(0.3)[3:], [0.0125, 0.03, 0.0125])  # the cross-set floor applies on top


def test_rule_is_unchanged_and_records_no_cv(tmp_path):
    stage = _stage(seed=1)
    s, info = cardvalue.select_sigma(stage, np.zeros(stage.prob.p), 20, 3, str(tmp_path), 1, seed=1, opts=cardvalue.ChainOpts("time", True))
    assert "theta" not in info
    fits = cardvalue.run_chain({"bo1": stage}, 20, 3, str(tmp_path), 1, seed=1, opts=cardvalue.ChainOpts("time", True))
    f = fits["bo1"]
    assert f.theta_cv_sd is None and f.spec.th_abs is None and f.theta_loss == []
    assert cardvalue.theta_sd_of(f) == pytest.approx(float(np.clip(4.0 * f.sigma, 0.05, 0.5)))
    # the fit is the old one: the penalised GLM with the rule's SD vector
    sig = np.concatenate([stage.enc.sigma, np.full(20, f.sigma), np.full(3, np.clip(4.0 * f.sigma, 0.05, 0.5))])
    ref = glm.fit(stage.prob, np.zeros(stage.prob.k + 23), sig, w=None, x0=np.zeros(stage.prob.k + 23), maxiter=800)
    np.testing.assert_array_equal(f.coef_m if f.coef_m is not None else f.coef, ref.coef)


def test_cv_picks_from_the_grid_records_it_and_follows_the_data(tmp_path):
    opts = cardvalue.ChainOpts("time", True, theta_sigma="cv")
    # (a) large true structure effects: a loose prior must win; (b) none: a tight one
    big = _stage(seed=2, n_groups=260, theta=(0.5, -0.5, 0.4))
    none = _stage(seed=3, n_groups=260, theta=(0.0, 0.0, 0.0))
    out = {}
    for tag, st in (("big", big), ("none", none)):
        d = tmp_path / tag
        d.mkdir()
        s, info = cardvalue.select_sigma(st, np.zeros(st.prob.p), 20, 3, str(d), 1, seed=1, opts=opts)
        th = info["theta"]
        assert th["grid"] == list(cardvalue.THETA_GRID) and len(th["loss"]) == len(th["se"]) == len(cardvalue.THETA_GRID)
        assert th["value"] in cardvalue.THETA_GRID and th["value"] == th["grid"][th["best"]]
        assert th["best"] == int(np.argmin(th["loss"]))
        assert len(info["w_loss"]) == len(cardvalue.STACK_GRID) or "w_loss" not in info      # stacking needs a prior stage (none here)
        out[tag] = th["value"]
    assert out["big"] > out["none"]
    assert out["big"] >= 0.1 and out["none"] <= 0.05


def test_cv_chain_records_theta_uses_it_in_the_fit_and_the_bootstrap_and_skips_the_draft(tmp_path):
    opts = cardvalue.ChainOpts("time", True, theta_sigma="cv")
    dr = _stage(seed=5, n_groups=200, theta=(0.3, 0.0, -0.3), name="draft")
    bo = _stage(seed=6, n_groups=200, theta=(0.3, 0.0, -0.3), name="bo1")
    stages = {"draft": dr, "bo1": bo}
    fits = cardvalue.run_chain(stages, 20, 3, str(tmp_path), 1, seed=1, opts=opts)
    d, f = fits["draft"], fits["bo1"]
    assert d.theta_cv_sd is None and d.spec.th_abs is None                          # the draft stage keeps the rule
    assert f.theta_cv_sd in cardvalue.THETA_GRID and f.spec.th_abs == f.theta_cv_sd
    assert f.theta_grid == list(cardvalue.THETA_GRID) and len(f.theta_loss) == len(f.theta_se) == len(cardvalue.THETA_GRID)
    assert cardvalue.theta_sd_of(f) == f.theta_cv_sd
    assert f.w_loss                                                                   # stacking ran after, at the chosen SDs
    k = f.k
    np.testing.assert_array_equal(f.spec.vector(f.sigma)[k + 20:k + 23], np.full(3, f.theta_cv_sd))
    # the plain fit M is the penalised GLM at exactly that vector
    m = f.coef_m if f.coef_m is not None else f.coef
    ref = glm.fit(bo.prob, cardvalue.prior_vector(k, f.prior_deck), f.spec.vector(f.sigma), w=None,
                  x0=cardvalue.prior_vector(k, f.prior_deck), maxiter=800)
    np.testing.assert_allclose(m, ref.coef, atol=1e-6)
    # the bootstrap replicates use the chosen value (the spec travels in the task)
    (tmp_path / "boot").mkdir()
    boot = cardvalue.bootstrap(stages, fits, 1, str(tmp_path / "boot"), 1, ["bo1"], seed=3)
    assert boot["bo1"].shape == (2, len(f.coef)) and np.all(np.isfinite(boot["bo1"][1]))
    # a model with extra blocks takes the card sigma, the stacking weight AND the structure SD from the plain chain's fit
    ex_dir = tmp_path / "ex"
    ex_dir.mkdir()
    sf = cardvalue._stage_fit(bo, f.prior_deck, 20, 3, str(ex_dir), 1, 1, None, None, lam=f.lam, lam_se=f.lam_se, lam_from=f.lam_from,
                              ref_fit=f, opts=opts)
    assert sf.theta_cv_sd == f.theta_cv_sd and sf.spec.th_abs == f.theta_cv_sd and sf.sigma == f.sigma and sf.w == f.w


def test_plumbing_of_the_option():
    assert inspect.signature(pipeline.calculate).parameters["theta_sigma"].default == "rule"
    from sealed_lab import api, cli
    assert inspect.signature(api.calculate_set).parameters["theta_sigma"].default == "rule"
    with pytest.raises(ValueError, match="theta_sigma"):
        pipeline.calculate("TST", theta_sigma="bogus")
    seen = {}

    def fake(set_code, **kw):
        seen.update(kw)
        raise SystemExit(0)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(api, "calculate_set", fake)
        for argv, want in ((["calc", "TST"], "rule"), (["calc", "TST", "--theta-sigma", "cv"], "cv")):
            seen.clear()
            with pytest.raises(SystemExit):
                cli.main(argv)
            assert seen["theta_sigma"] == want
        with pytest.raises(SystemExit):                                                # argparse rejects other values
            cli.main(["calc", "TST", "--theta-sigma", "auto"])
    finally:
        mp.undo()


# ------------------------------------------------------------------------------------------------ select_theta_w + the evaluation script

def test_select_theta_w_repeats_the_theta_step_of_select_sigma(tmp_path):
    opts = cardvalue.ChainOpts("time", True, theta_sigma="cv")
    st = _stage(seed=7, n_groups=200, theta=(0.4, 0.0, -0.4))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    s, info = cardvalue.select_sigma(st, np.zeros(st.prob.p), 20, 3, str(tmp_path / "a"), 1, seed=1, opts=opts)
    spec = cardvalue.make_spec(st, 20, 3, "bo1")
    spec2, info2 = cardvalue.select_theta_w(st, np.zeros(st.prob.p), spec, s, str(tmp_path / "b"), 1, seed=1, opts=opts, has_prior=False)
    np.testing.assert_allclose(info2["theta"]["loss"], info["theta"]["loss"], atol=1e-12)
    assert spec2.th_abs == info["theta"]["value"] and "w" not in info2                  # no prior stage: no stacking
    # a prior stage: the stacking weight is chosen too, at the chosen structure SD
    spec3, info3 = cardvalue.select_theta_w(st, np.full(st.prob.p, 0.02), spec, s, str(tmp_path / "b"), 1, seed=1, opts=opts)
    assert len(info3["w_loss"]) == len(cardvalue.STACK_GRID) and spec3.th_abs in cardvalue.THETA_GRID
    # the draft stage is never touched by the option (select_theta_w is explicit, select_sigma is by stage)
    dr = _stage(seed=7, n_groups=200, theta=(0.4, 0.0, -0.4), name="draft")
    (tmp_path / "c").mkdir()
    _s, info4 = cardvalue.select_sigma(dr, np.zeros(dr.prob.p), 20, 3, str(tmp_path / "c"), 1, seed=1, opts=opts)
    assert "theta" not in info4


def test_evaluation_script_helpers():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import split_half_decompose as shd
    import theta_sigma_check as tsc

    # verdict: the pipeline's confirmation rule on a material, significant first-half gain that holds in the second half
    gain = dict(gain=0.0005, lo=0.0001, hi=0.0009)
    halves = dict(A=dict(gain=0.0006, lo=0.0001, hi=0.0011, lo90=0.0002, hi90=0.001), B=dict(gain=0.0004, lo=-0.0002, hi=0.001, lo90=0.0001, hi90=0.0008))
    assert tsc.verdict_row(gain, halves, "bo3")["verdict"] == "better"
    halves["B"] = dict(gain=-0.0001, lo=-0.0005, hi=0.0003, lo90=-0.0004, hi90=0.0002)
    assert tsc.verdict_row(gain, halves, "bo3")["verdict"] != "better"
    # retarget_records: T_sm of the new coefficients = sum_j bA_j dphi_A - bB_j dphi_B (raw units b = theta / sd)
    names = shd.FEATURES[:3]
    sd = np.full(shd.NF, 2.0)
    ctx = {h: dict(sd=sd, b={}) for h in "AB"}
    theta = {"A": np.array([0.2, -0.1, 0.4]), "B": np.array([0.1, 0.1, 0.0])}
    refit = {h: dict(info=dict(names=names, formats=["bo3"]), arr={"bo3__theta": theta[h]}) for h in "AB"}
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(2, 2, shd.NF))
    rec = dict(set="TST", fmt="bo3", pool="p", T=0.1, s=0.25, phi=phi, T_sm=0.0, tot=np.zeros(shd.NF), bpart=np.zeros(shd.NF), ppart=np.zeros(shd.NF))
    [new] = tsc.retarget_records([rec], ctx, refit, "TST")
    bA = np.zeros(shd.NF)
    bB = np.zeros(shd.NF)
    bA[:3], bB[:3] = theta["A"] / 2.0, theta["B"] / 2.0
    want = bA @ (phi[0, 0] - phi[0, 1]) - bB @ (phi[1, 0] - phi[1, 1])
    assert new["T_sm"] == pytest.approx(want) and new["tot"].sum() == pytest.approx(want)
    assert new["bpart"].sum() + new["ppart"].sum() == pytest.approx(want)
    st = tsc.structure_stats([new], {})
    assert st["mean_pp"] == pytest.approx(50 * 0.25 * want)
