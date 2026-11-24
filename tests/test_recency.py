"""Recency weighting: weights, the auto/on/off decision rule, and its place in the end-to-end report."""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from sealed_lab import config
from sealed_lab.model import design, pipeline, validate


def _stage(starts: list[str], group: list[int]):
    return types.SimpleNamespace(group_start=np.array(starts, dtype="datetime64[us]"), group=np.array(group))


def test_recency_weights_halve_per_half_life():
    st = _stage(["2026-07-01", "2026-06-17", "2026-06-03"], [0, 0, 1, 2])
    w = design.recency_weights(st, np.datetime64("2026-07-01"), 14.0)
    assert np.allclose(w, [1.0, 1.0, 0.5, 0.25])
    assert design.recency_weights(st, np.datetime64("2026-07-01"), None) is None
    # events after the reference time are not up-weighted
    w2 = design.recency_weights(st, np.datetime64("2026-06-10"), 7.0)
    assert np.allclose(w2[:3], 1.0) and w2[3] == pytest.approx(0.5)


def _inner(gain_by_h: dict[float, float]) -> dict:
    rows = [dict(half_life=None, logloss=0.6, diff=0.0, lo=0.0, hi=0.0)]
    rows += [dict(half_life=h, logloss=0.6 - g, diff=g, lo=g - 0.01, hi=g + 0.01) for h, g in gain_by_h.items()]
    return {"bo1": dict(status="ok", rows=rows)}


def test_choose_half_life():
    assert validate.choose_half_life(_inner({60.0: 0.001, 30.0: 0.003, 14.0: -0.002})) == 30.0
    assert validate.choose_half_life(_inner({60.0: -0.001, 30.0: -0.003})) is None
    assert validate.best_finite_half_life(_inner({60.0: -0.001, 30.0: -0.003})) == 60.0
    assert validate.choose_half_life({"bo3": dict(status="skipped")}) is None


def _val(gain, lo, hi):
    return {"bo1": dict(status="ok", recency=dict(half_life=30.0, gain=gain, lo=lo, hi=hi))}


def test_decide_recency_rules():
    # the whole-hold-out rule (no half-split numbers here: with the confirmation split a format without halves cannot pass)
    inner = _inner({30.0: 0.003})
    on = pipeline.decide_recency("auto", 30.0, inner, _val(0.004, 0.001, 0.007), confirm=False)
    assert on["applied"] and on["half_life"] == 30.0
    unsure = pipeline.decide_recency("auto", 30.0, inner, _val(0.002, -0.001, 0.005), confirm=False)
    assert not unsure["applied"] and "includes" in unsure["reason"]
    worse = pipeline.decide_recency("auto", 30.0, inner, _val(-0.004, -0.007, -0.001), confirm=False)
    assert not worse["applied"] and "WORSE" in worse["reason"]
    none = pipeline.decide_recency("auto", None, _inner({30.0: -0.001}), {})
    assert not none["applied"]
    forced = pipeline.decide_recency("on", None, _inner({30.0: -0.001, 14.0: -0.004}), {})
    assert forced["applied"] and forced["half_life"] == 30.0
    off = pipeline.decide_recency("off", 30.0, inner, _val(0.004, 0.001, 0.007))
    assert not off["applied"]


def test_report_has_recency_section(calc):
    rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    rc = rep["recency"]
    assert rc["mode"] == "auto" and isinstance(rc["applied"], bool) and rc["reason"]
    for fmt in ("bo3", "bo1"):
        inner = rc["inner"][fmt]
        assert inner["status"] == "ok"
        assert [r["half_life"] for r in inner["rows"]] == [None, *validate.RECENCY_GRID]
        assert all(r["lo"] <= r["diff"] <= r["hi"] for r in inner["rows"][1:])
    assert rc["scope"] == "format" and set(rc["chosen"]) == {"bo3", "bo1"}      # one half-life per format
    for fmt in ("bo3", "bo1"):
        if rc["chosen"][fmt] is not None:   # the hold-out test of this format ran: MR is in its metrics and summarised
            v = rep["validation"][fmt]
            assert "MR" in v["metrics"] and v["recency"]["lo"] <= v["recency"]["gain"] <= v["recency"]["hi"]
    meta = json.loads((config.model_dir("TST") / "meta.json").read_text(encoding="utf-8"))
    assert meta["recency"]["applied"] == rc["applied"]
    html = (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    assert "Recency weighting" in html and ("APPLIED" in html)


def test_negligible_significant_differences_neither_apply_nor_veto():
    # a near-identical model: CI excludes zero but the effect is 1e-5 log loss per game (seen with the cross-set prior)
    tiny_gain = {"bo1": dict(status="ok", xs=dict(gain=1.7e-5, lo=1.2e-5, hi=2.2e-5))}
    d = pipeline.decide_component("auto", "xs", tiny_gain, True, "", "the prior", confirm=False)
    assert not d["applied"] and "materiality floor" in d["reason"]
    tiny_loss = {"bo1": dict(status="ok", xs=dict(gain=-5e-6, lo=-7e-6, hi=-3e-6)),
                 "bo3": dict(status="ok", xs=dict(gain=0.002, lo=0.0005, hi=0.0035))}
    d = pipeline.decide_component("auto", "xs", tiny_loss, True, "", "the prior", confirm=False)
    assert d["applied"]                      # a negligible loss in bo1 does not veto a material gain in bo3


def test_deployed_combination_is_checked(calc):
    rep = json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))
    dp = rep["deployed"]
    assert isinstance(dp["components"], list) and isinstance(dp["fallback"], bool)
    for name in dp["components"]:
        assert rep[name]["applied"] or dp["fallback"]
    for fmt, o in dp["formats"].items():
        if o["status"] == "ok":
            assert o["lo"] <= o["gain"] <= o["hi"] and set(o["components"]) == set(dp["components"])
    html = (config.model_dir("TST") / "report.html").read_text(encoding="utf-8")
    assert "The deployed model as a whole" in html
