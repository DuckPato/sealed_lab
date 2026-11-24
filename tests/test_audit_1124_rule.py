"""Audit 2026-11-24 C1/C2: the shipping rule fails closed (a format without half numbers cannot pass; an untested format blocks a shared component)."""
from sealed_lab.model import pipeline


def _o(gain=0.001, lo=0.0005, hi=0.0015, halves=True):
    o = dict(gain=gain, lo=lo, hi=hi)
    if halves:
        o["halves"] = dict(A=dict(gain=gain, lo=lo, hi=hi, lo90=lo), B=dict(gain=gain, lo=lo, hi=hi, lo90=lo))
    return o


def _val(**fmts):
    return {f: ({"status": "ok", "comp": o} if o else {"status": "too few events"}) for f, o in fmts.items()}


def test_c1_format_without_halves_cannot_pass():
    ok = pipeline._verdict({"bo3": _o(), "bo1": _o()}, True)
    assert ok[0] == ["bo3", "bo1"]
    better, worse, _t, info = pipeline._verdict({"bo3": _o(), "bo1": _o(halves=False)}, True)
    assert better == ["bo3"] and info["no_halves"] == ["bo1"] and info["confirm"]       # bo3 still judged normally
    better, _w, _t, _i = pipeline._verdict({"bo1": _o(halves=False)}, True)
    assert better == []                                                                  # whole-hold-out pass alone is not enough
    assert pipeline._verdict({"bo1": _o(halves=False)}, False)[0] == ["bo1"]            # no confirmation asked: old rule


def test_c1_worse_still_judged_without_halves():
    bad = _o(-0.001, -0.0015, -0.0005, halves=False)
    _b, worse, _t, _i = pipeline._verdict({"bo3": _o(), "bo1": bad}, True)
    assert worse == ["bo1"]
    d = pipeline.decide_component("auto", "comp", _val(bo3=_o(), bo1=bad), True, "", "x")
    assert not d["applied"]


def test_c1_decision_not_applied_when_only_whole_holdout_passes():
    d = pipeline.decide_component("auto", "comp", _val(bo3=_o(halves=False)), True, "", "x")
    assert not d["applied"] and "cannot be confirmed" in d["reason"]


def test_c2_untested_format_blocks_shared_component():
    d = pipeline.decide_component("auto", "comp", _val(bo3=_o(), bo1=None), True, "", "x")
    assert not d["applied"] and d["untested"] == ["bo1"] and "not tested in bo1" in d["reason"]
    assert pipeline.decide_component("auto", "comp", _val(bo3=_o(), bo1=_o()), True, "", "x")["applied"]
    assert pipeline.decide_component("on", "comp", _val(bo3=_o(), bo1=None), True, "", "x")["applied"]      # forced on: the user's call
    # per-format components (the matchup layer) are not blocked
    assert pipeline.decide_component("auto", "comp", _val(bo3=_o(), bo1=None), True, "", "x", shared=False)["applied"]


def test_c2_castability_untested_format_not_applied():
    v = {"bo3": {"status": "ok", "castability": _o(0.0, -0.0001, 0.0001)}, "bo1": {"status": "too few"}}
    d = pipeline.decide_castability("auto", v, True)
    assert not d["applied"] and d["untested"] == ["bo1"]
    v["bo1"] = {"status": "ok", "castability": _o(0.0, -0.0001, 0.0001)}
    assert pipeline.decide_castability("auto", v, True)["applied"]


def test_c2_extra_draft_untested_format():
    ev = {"PickTwoDraft": 100}
    t = {"bo3": _o()}
    assert pipeline.decide_extra_draft("auto", ev, 1000, t, True)["applied"]                       # old callers: unchanged
    d = pipeline.decide_extra_draft("auto", ev, 1000, t, True, ["bo3", "bo1"])
    assert not d["applied"] and d["untested"] == ["bo1"]
    assert pipeline.decide_extra_draft("auto", ev, 1000, {"bo3": _o(), "bo1": _o()}, True, ["bo3", "bo1"])["applied"]
