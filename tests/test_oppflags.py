"""Opponent color flags in the base covariates (docs/opp_flags/RULE.md): the design columns, the reference convention, off == the
previous behaviour, the decision (per-set rule + pooled route that only switches ON) and one end-to-end run on the synthetic set."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from sealed_lab.model import design, pipeline


def _games(opp) -> pl.DataFrame:
    n = len(opp)
    t0 = dt.datetime(2025, 1, 1)
    return pl.DataFrame(dict(
        event=["Sealed"] * n, game_time=[t0 + dt.timedelta(days=i) for i in range(n)], skill=[0.55] * n, on_play=[bool(i % 2) for i in range(n)],
        num_mulligans=[0] * n, opp_num_mulligans=[0] * n, game_number=[1] * n, opp_colors=opp, won=[bool(i % 2) for i in range(n)],
        draft_id=[f"d{i // 2}" for i in range(n)], ))


OPP = ["WU", "G", "", None, "WUBRG", "RG", "xyz", "BR"]


def test_flag_matrix_rows():
    F = design.opp_flag_matrix(OPP)
    assert F.shape == (8, 6)
    assert F[0].tolist() == [1, 1, 0, 0, 0, 2]          # WU
    assert F[1].tolist() == [0, 0, 0, 0, 1, 1]          # G
    assert F[2].tolist() == [0] * 6 and F[3].tolist() == [0] * 6          # empty and null -> all 0, zero colors
    assert F[4].tolist() == [1, 1, 1, 1, 1, 5]
    assert F[5].tolist() == [0, 0, 0, 1, 1, 2]          # RG
    assert F[6].tolist() == [0] * 6                      # no color letters
    assert F[7].tolist() == [0, 0, 1, 1, 0, 2]          # BR (order of the string does not matter)


def test_encoder_off_is_unchanged_and_on_appends_columns_after_the_pair_one_hot():
    g = _games(OPP)
    e0 = design.CovEncoder.fit(g)
    e1 = design.CovEncoder.fit(design.with_opp_flags(g))
    assert not e0.oppflags and e1.oppflags
    assert e1.names[:e0.k] == e0.names and e1.names[e0.k:] == design.OPP_FLAG_NAMES
    assert e0.names[-len(design.OPP_LEVELS):] == [f"opp:{o}" for o in design.OPP_LEVELS]
    assert not any(n.startswith("oppflag") for n in e0.names)
    X0, X1 = e0.transform(g), e1.transform(design.with_opp_flags(g))
    assert X1.shape == (g.height, e0.k + 6)
    assert (X1[:, :e0.k] != X0).nnz == 0                 # the old columns are untouched
    assert np.allclose(X1[:, e0.k:].toarray(), design.opp_flag_matrix(OPP))
    assert np.all(e1.sigma[:e0.k] == e0.sigma) and len(e1.sigma) == e1.k


def test_reference_row_averages_the_flags_over_the_observed_mix():
    g = design.with_opp_flags(_games(OPP))
    enc = design.CovEncoder.fit(g)
    C = enc.transform(g)
    ref = enc.set_reference(C)
    ix = {n: i for i, n in enumerate(enc.names)}
    F = design.opp_flag_matrix(OPP)
    for j, nm in enumerate(design.OPP_FLAG_NAMES):
        assert ref[ix[nm]] == pytest.approx(F[:, j].mean())
    assert ref[ix["on_play"]] == 0.5 and ref[ix["intercept"]] == 1.0       # the pilot-neutral convention is unchanged


def test_marker_is_idempotent_and_does_not_touch_the_input():
    g = _games(OPP)
    g1 = design.with_opp_flags(g)
    assert design.OPP_FLAG_MARK in g1.columns and design.OPP_FLAG_MARK not in g.columns
    assert design.with_opp_flags(g1) is g1


def test_mode_is_validated():
    with pytest.raises(ValueError, match="oppflags"):
        pipeline.calculate("TST", oppflags="maybe")


# ------------------------------------------------------------------------------------------------ the decision
def _entry(gain, lo, hi, a=None, b=None):
    a = a or dict(gain=gain, lo=lo, hi=hi, lo90=lo)
    b = b or dict(gain=gain, lo=lo, hi=hi, lo90=lo)
    return dict(gain=gain, lo=lo, hi=hi, halves=dict(A=a, B=b))


def _val(**per_fmt):
    return {f: dict(status="ok", oppflags=e) for f, e in per_fmt.items()}


def test_decide_oppflags_auto_applies_on_a_confirmed_gain(tmp_path):
    d = pipeline.decide_oppflags("auto", _val(bo1=_entry(2e-4, 5e-5, 3e-4), bo3=_entry(0.0, -1e-4, 1e-4)), ["bo1", "bo3"], "TST",
                                 others={"bo1": {}, "bo3": {}})
    assert d["applied"] and d["source"] == "per-set"


def test_decide_oppflags_auto_does_not_apply_a_tiny_or_unconfirmed_gain():
    tiny = _entry(5e-5, 1e-5, 9e-5)
    d = pipeline.decide_oppflags("auto", _val(bo1=tiny), ["bo1"], "TST", others={"bo1": {}})
    assert not d["applied"]
    b_fail = dict(gain=-1e-4, lo=-5e-4, hi=1e-4, lo90=-3e-4)
    d = pipeline.decide_oppflags("auto", _val(bo1=_entry(3e-4, 1e-4, 5e-4, b=b_fail)), ["bo1"], "TST", others={"bo1": {}})
    assert not d["applied"] and "NOT confirmed" in d["reason"]


def test_decide_oppflags_blocks_a_material_loss_and_honours_on_off():
    bad = _entry(-3e-4, -5e-4, -1e-4)
    assert not pipeline.decide_oppflags("auto", _val(bo1=bad), ["bo1"], "TST", others={"bo1": {}})["applied"]
    assert pipeline.decide_oppflags("on", _val(bo1=bad), ["bo1"], "TST")["applied"]
    off = pipeline.decide_oppflags("off", _val(bo1=_entry(2e-4, 5e-5, 3e-4)), ["bo1"], "TST")
    assert not off["applied"] and off["mode"] == "off"


def test_pooled_route_only_switches_on():
    cell = lambda: dict(whole=dict(gain=3e-4, lo=1e-4, hi=5e-4), A=dict(gain=3e-4, lo=1e-4, hi=5e-4, lo90=1.5e-4),   # noqa: E731
                        B=dict(gain=3e-4, lo=1e-4, hi=5e-4, lo90=1.5e-4))
    others = {"bo1": {f"S{i}": cell() for i in range(6)}}
    flat = _entry(0.0, -1e-4, 1e-4)                                  # own numbers: no per-set gain, not materially worse
    d = pipeline.decide_oppflags("auto", _val(bo1=flat), ["bo1"], "TST", others=others)
    assert d["applied"] and d["source"] == "pooled" and "pooled" in d
    worse = _entry(-3e-4, -5e-4, -1e-4)                              # materially worse own numbers block it
    assert not pipeline.decide_oppflags("auto", _val(bo1=worse), ["bo1"], "TST", others=others)["applied"]
    neg = {"bo1": {f"S{i}": dict(whole=dict(gain=-3e-4, lo=-5e-4, hi=-1e-4), A=dict(gain=-3e-4, lo=-5e-4, hi=-1e-4, lo90=-4e-4),
                                B=dict(gain=-3e-4, lo=-5e-4, hi=-1e-4, lo90=-4e-4)) for i in range(6)}}
    good = _entry(2e-4, 5e-5, 3e-4)
    assert pipeline.decide_oppflags("auto", _val(bo1=good), ["bo1"], "TST", others=neg)["applied"]     # per-set pass is never switched off


# ------------------------------------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The synthetic set calculated with oppflags off and on, each in its own scratch home."""
    from sealed_lab import api
    from sealed_lab.ingest.corpus import build_corpus
    from sealed_lab.model.artifact import SetModel
    from sealed_lab.synth.generate import generate

    out = {}
    mp = pytest.MonkeyPatch()
    synth = tmp_path_factory.mktemp("opp_synth")
    generate(synth, set_code="TST", seed=0, scale=0.3)
    try:
        for mode in ("off", "on"):
            home = tmp_path_factory.mktemp(f"opp_home_{mode}")
            mp.setenv("SEALED_LAB_HOME", str(home))
            api.add_files([str(synth)])
            build_corpus("TST", n_jobs=2, fetch_scryfall=False)
            meta = pipeline.calculate("TST", n_jobs=2, n_boot=4, oppflags=mode)
            m = SetModel.load("TST")
            out[mode] = dict(meta=meta, beta={f: np.array(m.card_beta(f)) for f in meta["formats"]})
    finally:
        mp.undo()
    return out


def test_off_records_the_decision_and_has_no_flag_validation(runs):
    d = runs["off"]["meta"]["oppflags"]
    assert d["mode"] == "off" and not d["applied"] and "switched off" in d["reason"]


def test_on_runs_end_to_end_and_records_its_decision(runs):
    meta = runs["on"]["meta"]
    d = meta["oppflags"]
    assert d["mode"] == "on" and d["applied"] and d["source"] == "forced"
    assert "oppflags" in meta["deployed"]["components"] or meta["deployed"]["fallback"]
    for f, b in runs["on"]["beta"].items():
        assert np.all(np.isfinite(b))
    # the flags move the card values (a different base), so the fits are not the plain ones
    assert any(not np.allclose(runs["on"]["beta"][f], runs["off"]["beta"][f]) for f in runs["on"]["beta"]) or meta["deployed"]["fallback"]


def test_default_is_off(calc):
    """The default is "off" (docs/opp_flags/RESULT.md: opp_colors are the colors shown during the game): `calc` runs the default."""
    d = calc["meta"]["oppflags"]
    assert d["mode"] == "off" and not d["applied"] and "reason" in d
