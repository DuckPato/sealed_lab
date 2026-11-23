"""scripts/extra_draft_check.py: the pure parts (registry plan, SD, hold-out alignment, paired gain, verdict) on synthetic numbers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("extra_draft_check", _SCRIPTS / "extra_draft_check.py")
ed = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ed)

ENTRIES = [
    dict(kind="game", event="Sealed", fingerprint="s"), dict(kind="game", event="PickTwoDraft", fingerprint="g2"),
    dict(kind="draft", event="PickTwoDraft", fingerprint="d2"), dict(kind="replay", event="PickTwoDraft", fingerprint="r2"),
    dict(kind="game", event="PremierDraft", fingerprint="gp"), dict(kind="draft", event="PremierDraft", fingerprint="dp"),
]


def test_registry_plan_removal_keeps_replay_and_premier():
    rm, add = ed.registry_plan(ENTRIES, "FRA", (), True)
    assert sorted(rm) == ["d2", "g2"] and add == []


def test_registry_plan_add_skips_registered_and_names_game_files():
    rm, add = ed.registry_plan(ENTRIES, "OM1", ("PickTwoDraft", "PickTwoTradDraft"), False)
    assert rm == [] and len(add) == 1 and add[0].endswith("game_data_public.OM1.PickTwoTradDraft.csv.gz")
    _rm, add2 = ed.registry_plan([ENTRIES[0]], "OM1", ("PickTwoDraft",), False)
    assert add2[0].endswith("game_data_public.OM1.PickTwoDraft.csv.gz")


def test_weighted_sd():
    x = np.array([1.0, 3.0, 100.0])
    assert ed.weighted_sd(x[:2]) == pytest.approx(1.0)
    assert ed.weighted_sd(x, [1, 1, 0]) == pytest.approx(1.0)            # zero weight ignored
    assert ed.weighted_sd(x, [3, 1, 0]) == pytest.approx(np.sqrt(0.75))
    assert np.isnan(ed.weighted_sd([], None))


def test_align_holdouts_orders_and_detects_differences():
    a = dict(did=np.array(["b", "a", "b", "a"]), y=np.array([1, 0, 0, 1]))
    b = dict(did=np.array(["a", "b", "a", "b"]), y=np.array([0, 1, 1, 0]))      # same games, a different arrival order
    ia, ib, g = ed.align_holdouts(a, b)
    assert list(g) == [0, 0, 1, 1]
    assert np.array_equal(a["y"][ia], b["y"][ib])
    with pytest.raises(ValueError):
        ed.align_holdouts(a, dict(did=np.array(["a", "b", "a"]), y=np.array([0, 1, 1])))
    with pytest.raises(ValueError):
        ed.align_holdouts(a, dict(did=b["did"], y=np.array([0, 1, 0, 0])))


def _sim(rng, n_ev=300, per=8, gain=0.0):
    g = np.repeat(np.arange(n_ev), per)
    p = rng.uniform(0.3, 0.7, n_ev * per)
    y = (rng.uniform(size=len(p)) < p).astype(float)
    p_without = np.clip(p + rng.normal(0, 0.15, len(p)), 0.02, 0.98)
    p_with = p_without if gain == 0 else p                                    # "with" = the true probabilities
    return y, p_without, p_with, g


def test_paired_gain_sign_and_halves():
    rng = np.random.default_rng(0)
    y, pwo, pw, g = _sim(rng, gain=1.0)
    first = np.arange(300) < 150
    out = ed.paired_gain(y, pwo, pw, g, first, 200, 1)
    assert out["gain"] > 0 and out["lo"] <= out["gain"] <= out["hi"]
    assert out["logloss_without"] - out["logloss_with"] == pytest.approx(out["gain"])
    assert out["halves"]["A"]["n_events"] == 150 and out["halves"]["B"]["n_events"] == 150
    same = ed.paired_gain(y, pwo, pwo, g, first, 200, 1)
    assert same["gain"] == 0 and same["lo"] == 0 and same["hi"] == 0


def test_shipping_verdict_rule():
    def half(gain, lo, hi, lo90):
        return dict(gain=gain, lo=lo, hi=hi, lo90=lo90)

    ok = dict(gain=0.001, lo=0.0005, hi=0.0015, halves=dict(A=half(0.001, 0.0004, 0.0016, 0.0006), B=half(0.001, 0.0, 0.002, 0.0003)))
    v = ed.shipping_verdict({"bo1": ok})
    assert v["ship"] and v["better"] == ["bo1"]
    unconfirmed = dict(ok, halves=dict(A=ok["halves"]["A"], B=half(-0.0005, -0.002, 0.001, -0.001)))
    v = ed.shipping_verdict({"bo1": unconfirmed})
    assert not v["ship"] and "not confirmed" in v["text"]
    worse = dict(gain=-0.001, lo=-0.002, hi=-0.0005, halves=dict(A=half(-0.001, -0.002, -0.0004, -0.002), B=half(-0.001, -0.002, -0.0004, -0.002)))
    v = ed.shipping_verdict({"bo1": ok, "bo3": worse})
    assert not v["ship"] and v["worse"] == ["bo3"]
    flat = dict(gain=0.0, lo=-0.001, hi=0.001, halves=dict(A=half(0, -0.001, 0.001, -0.0005), B=half(0, -0.001, 0.001, -0.0005)))
    v = ed.shipping_verdict({"bo1": flat})
    assert not v["ship"] and "no significant gain" in v["text"]
