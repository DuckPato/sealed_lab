"""The "pooled" matchups mode: the decision pooled over the other sets' stored hold-out numbers (model/pooling.py,
pipeline.decide_matchups), the per-format application of the layer, and the plumbing of the new mode.  Synthetic numbers only: no
model is fitted, no real app home is read."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from sealed_lab.deck import matchups as dm
from sealed_lab.model import artifact, pipeline, pooling


def _o(gain, se):
    return dict(gain=gain, lo=gain - 1.96 * se, hi=gain + 1.96 * se, lo90=gain - 1.2816 * se)


def _cell(w, a, b, se=0.0005):
    return dict(whole=_o(w, se), A=_o(a, se), B=_o(b, se))


def _good(n=6, gain=0.002, se=0.0005, tag="S"):
    return {f"{tag}{i}": _cell(gain, gain, gain, se) for i in range(n)}


def _val(w, a, b, se=0.0005, status="ok"):
    """val[fmt] with an 'opponent' test the way run_validation stores it (whole hold-out + halves)."""
    o = _o(w, se)
    o["halves"] = dict(A=_o(a, se), B=_o(b, se))
    o.update(mode="pairs", sigma=0.01)
    return dict(status=status, opponent=o)


NEUTRAL = (0.00002, 0.00002, 0.00002, 0.005)          # (w, a, b, se): CI includes zero, below MIN_GAIN: the per-set rule says no


# ------------------------------------------------------------------------------------------------ pooled_decision
def test_pooled_better_applies():
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": _good(6)})["bo3"]
    assert d["decided"] and d["apply"] and d["verdict"]["label"] == "pooled-better" and d["n_sets"] == 6
    assert d["pooled"]["A"]["lo"] > 0


def test_not_better_does_not_apply():
    weak = {f"S{i}": _cell(0.0001, 0.0001, 0.0001, 0.005) for i in range(8)}           # CI wide around a tiny gain
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": weak})["bo3"]
    assert d["decided"] and not d["apply"] and d["verdict"]["label"] == "neither"


def test_half_b_must_confirm():
    oth = {f"S{i}": _cell(0.001, 0.002, -0.0005, 0.0005) for i in range(6)}             # half A fine, half B reversed
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": oth})["bo3"]
    assert d["verdict"]["better"] is False and not d["apply"]


def test_pooled_worse_blocks():
    oth = _good(6)
    oth.update({f"W{i}": _cell(-0.01, 0.002, 0.002, 0.0005) for i in range(20)})       # whole hold-out clearly worse in many sets
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": oth})["bo3"]
    assert d["verdict"]["worse"] and not d["apply"]


@pytest.mark.parametrize("where", ["whole", "A", "B"])
def test_own_materially_worse_blocks(where):
    own = _cell(0.0, 0.0, 0.0, 0.005)
    own[where] = _o(-0.002, 0.0005)                                                     # CI below zero and gain <= -MIN_GAIN
    d = pooling.pooled_decision({"bo3": own}, {"bo3": _good(6)})["bo3"]
    assert d["decided"] and d["own_worse"] and d["verdict"]["better"] and not d["apply"]


def test_own_slightly_negative_is_not_materially_worse():
    own = _cell(-0.0003, -0.0003, -0.0003, 0.005)                                       # CI includes zero
    assert pooling.pooled_decision({"bo3": own}, {"bo3": _good(6)})["bo3"]["apply"]


def test_fewer_than_five_other_sets_falls_back():
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": _good(4)})["bo3"]
    assert not d["decided"] and not d["apply"] and "4 other set" in d["why"]
    assert pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": _good(5)})["bo3"]["apply"]


def test_own_set_is_excluded_from_its_own_verdict():
    others = _good(4)                                         # only 4 real others ...
    others["TST"] = _cell(0.01, 0.01, 0.01, 0.0005)           # ... plus the set itself, which would make 5 and a great verdict
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": others}, exclude="tst")["bo3"]
    assert d["n_sets"] == 4 and "TST" not in d["sets"] and not d["decided"]
    others.update(_good(1, tag="X"))
    d = pooling.pooled_decision({"bo3": _cell(0.0, 0.0, 0.0, 0.005)}, {"bo3": others}, exclude="TST")["bo3"]
    assert d["n_sets"] == 5 and d["pooled"]["whole"]["mean"] == pytest.approx(0.002)        # the 0.01 of the set itself is not in it


def test_no_own_numbers_is_not_decided():
    d = pooling.pooled_decision({"bo3": None}, {"bo3": _good(6)})["bo3"]
    assert not d["decided"] and not d["apply"]


def test_cell_from_report_and_other_cells_skip_incomplete(tmp_path):
    def report(code, w, a, b, status="ok", halves=True):
        rep = dict(validation={"bo3": dict(status=status, opponent=_o(w, 0.0005)), "bo1": dict(status="ok", opponent=_o(w, 0.0005))},
                   matchups=dict(confirmation=dict(halves={"bo3": dict(A=_o(a, 0.0005), B=_o(b, 0.0005))} if halves else {})))
        d = tmp_path / "sets" / code / "model"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")

    for i in range(3):
        report(f"A{i}", 0.002, 0.002, 0.002)
    report("OWN", 0.5, 0.5, 0.5)
    report("NOHALVES", 0.002, 0.002, 0.002, halves=False)
    report("BADSTATUS", 0.002, 0.002, 0.002, status="skipped")
    (tmp_path / "sets" / "BROKEN" / "model").mkdir(parents=True)
    (tmp_path / "sets" / "BROKEN" / "model" / "report.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "sets" / "EMPTY" / "model").mkdir(parents=True)
    got = pooling.other_cells("own", "matchups", "opponent", ("bo1", "bo3"), tmp_path)
    assert sorted(got["bo3"]) == ["A0", "A1", "A2"]           # the set itself, the incomplete and the unreadable ones are left out
    assert got["bo1"] == {}                                   # no halves stored for bo1 in any report
    assert got["bo3"]["A0"]["A"]["gain"] == pytest.approx(0.002)


# ------------------------------------------------------------------------------------------------ decide_matchups
def test_per_set_rule_that_applies_is_kept_and_pool_is_not_used():
    val = {"bo1": _val(0.003, 0.003, 0.003), "bo3": _val(0.003, 0.003, 0.003)}
    d = pipeline.decide_matchups("pooled", val, ["bo1", "bo3"], "TST", others={"bo1": {}, "bo3": {}})
    assert d["applied"] and d["applied_formats"] == ["bo1", "bo3"] and d["source"] == {"bo1": "per-set", "bo3": "per-set"}
    assert "pooled" not in d and d["mode"] == "pooled"


def test_pooled_applies_bo3_but_not_bo1():
    """FRA's case: nothing passes per set; the pooled evidence carries bo3, bo1 has too few other sets."""
    val = {"bo1": _val(*NEUTRAL), "bo3": _val(*NEUTRAL)}
    others = {"bo1": _good(3), "bo3": _good(7)}
    d = pipeline.decide_matchups("pooled", val, ["bo1", "bo3"], "FRA", others=others)
    assert d["applied"] and d["applied_formats"] == ["bo3"] and d["source"] == {"bo3": "pooled"}
    assert d["pooled"]["bo3"]["apply"] and d["pooled"]["bo3"]["n_sets"] == 7 and d["pooled"]["bo3"]["verdict"] == "pooled-better"
    assert not d["pooled"]["bo1"]["decided"] and d["pooled"]["bo1"]["n_sets"] == 3
    assert set(d["pooled"]["bo3"]["re"]) == {"whole", "A", "B"} and "tau2" in d["pooled"]["bo3"]["re"]["whole"]
    assert "pooled across sets" in d["reason"] and "bo3" in d["reason"] and d["per_set_reason"]


def test_pooled_own_materially_worse_blocks_that_format_only():
    val = {"bo1": _val(-0.003, -0.003, -0.003), "bo3": _val(*NEUTRAL)}
    d = pipeline.decide_matchups("pooled", val, ["bo1", "bo3"], "TST", others={"bo1": _good(6), "bo3": _good(6)})
    assert d["applied_formats"] == ["bo3"] and d["pooled"]["bo1"]["own_worse"] and not d["pooled"]["bo1"]["apply"]


def test_pooled_not_better_leaves_the_per_set_result():
    val = {"bo3": _val(*NEUTRAL)}
    weak = {f"S{i}": _cell(0.0001, 0.0001, 0.0001, 0.005) for i in range(8)}
    d = pipeline.decide_matchups("pooled", val, ["bo3"], "TST", others={"bo3": weak})
    assert not d["applied"] and d["applied_formats"] == [] and d["source"] == {}
    assert d["reason"].startswith(d["per_set_reason"]) and "evidence pooled across sets" in d["reason"]


def test_pooled_with_too_few_sets_is_the_per_set_auto_result():
    val = {"bo1": _val(*NEUTRAL), "bo3": _val(*NEUTRAL)}
    pooled = pipeline.decide_matchups("pooled", val, ["bo1", "bo3"], "TST", others={"bo1": _good(2), "bo3": _good(4)})
    auto = pipeline.decide_matchups("auto", val, ["bo1", "bo3"], "TST")
    assert not pooled["applied"] and not auto["applied"] and pooled["applied_formats"] == auto["applied_formats"] == []


def test_auto_on_off_do_not_read_the_other_sets():
    val = {"bo1": _val(*NEUTRAL), "bo3": _val(*NEUTRAL)}
    boom = {"bo1": _good(9), "bo3": _good(9)}
    assert not pipeline.decide_matchups("auto", val, ["bo1", "bo3"], "TST", others=boom)["applied"]
    off = pipeline.decide_matchups("off", val, ["bo1", "bo3"], "TST", others=boom)
    assert not off["applied"] and off["applied_formats"] == []
    on = pipeline.decide_matchups("on", val, ["bo1", "bo3"], "TST", others=boom)
    assert on["applied"] and on["applied_formats"] == ["bo1", "bo3"] and on["source"] == {"bo1": "forced", "bo3": "forced"}


def test_pooled_reads_the_app_home_when_others_not_given(tmp_path, monkeypatch):
    for i in range(5):
        d = tmp_path / "sets" / f"S{i}" / "model"
        d.mkdir(parents=True)
        rep = dict(validation={"bo3": dict(status="ok", opponent=_o(0.002, 0.0005))},
                   matchups=dict(confirmation=dict(halves={"bo3": dict(A=_o(0.002, 0.0005), B=_o(0.002, 0.0005))})))
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    d = pipeline.decide_matchups("pooled", {"bo3": _val(*NEUTRAL)}, ["bo3"], "TST")
    assert d["applied_formats"] == ["bo3"] and d["source"] == {"bo3": "pooled"}


# ------------------------------------------------------------------------------------------- per-format application
class _FakeOpp:
    mode = "pairs"
    opps = ["WU"]


def _model(applied_formats, applied=True):
    return SimpleNamespace(opp=_FakeOpp(), opp_formats=list(applied_formats), meta=dict(matchups=dict(mode="pooled", applied=applied)))


def test_has_opp_per_format():
    m = _model(["bo3"])
    assert artifact.SetModel.has_opp(m, "bo3") and not artifact.SetModel.has_opp(m, "bo1")
    assert not artifact.SetModel.has_opp(SimpleNamespace(opp=None, opp_formats=["bo3"]), "bo3")


def test_plan_is_not_supported_in_a_format_the_layer_is_not_applied_in():
    plan = dm.plan_for(_model(["bo1"]), {1: 1}, {1: 1}, "bo3")
    assert plan.status == "not_supported" and "not applied in this format" in plan.message and "bo1" in plan.message


def test_plan_in_bo1_stays_off():
    assert dm.plan_for(_model(["bo3"]), {1: 1}, {1: 1}, "bo1").status == "off"


def test_old_models_without_applied_formats_keep_all_formats():
    """meta.matchups without `applied_formats` (calculated before the pooled mode) = applied in every format."""
    for meta, expect in ((dict(applied=True), ["bo1", "bo3"]), (dict(applied=True, applied_formats=["bo3"]), ["bo3"])):
        stub = SimpleNamespace(meta=dict(matchups=meta), formats=["bo1", "bo3"])
        mm = stub.meta.get("matchups") or {}
        got = list(mm["applied_formats"]) if "applied_formats" in mm else list(stub.formats)
        assert got == expect                                   # (the expression of SetModel._setup_extras)
    src = inspect.getsource(artifact.SetModel._setup_extras)
    assert 'mm["applied_formats"]' in src and "else list(self.formats)" in src


# ------------------------------------------------------------------------------------------------------ plumbing
def test_default_and_modes():
    assert pipeline.MATCHUPS_MODES[0] == "pooled" and set(pipeline.MATCHUPS_MODES) == {"pooled", "auto", "on", "off"}
    assert inspect.signature(pipeline.calculate).parameters["matchups"].default == "pooled"
    from sealed_lab import api
    from sealed_lab.gui import fake_api
    assert inspect.signature(api.calculate_set).parameters["matchups"].default == "pooled"
    assert inspect.signature(fake_api.calculate_set).parameters["matchups"].default == "pooled"


def test_cli_matchups_flag(monkeypatch):
    from sealed_lab import api, cli

    seen = []

    def fake_calc(set_code, **kw):
        seen.append(kw["matchups"])
        return SimpleNamespace()

    monkeypatch.setattr(api, "calculate_set", fake_calc)
    monkeypatch.setattr(cli, "_print_status", lambda st: None)
    for args, want in (([], "pooled"), (["--matchups", "auto"], "auto"), (["--matchups", "pooled"], "pooled"),
                       (["--matchups", "on"], "on"), (["--matchups", "off"], "off")):
        assert cli.main(["calc", "TST"] + args) == 0
        assert seen[-1] == want
    with pytest.raises(SystemExit):
        cli.main(["calc", "TST", "--matchups", "maybe"])


def test_calculate_rejects_an_unknown_matchups_mode():
    with pytest.raises(ValueError, match="matchups must be one of"):
        pipeline.calculate("NOPE", matchups="maybe")


def test_report_renders_the_pooled_section():
    from sealed_lab.model import validate

    val = {"bo3": dict(status="ok", opponent=dict(_o(0.0, 0.005), auc_gain=0.0, auc_lo=-0.01, auc_hi=0.01, mode="pairs", sigma=0.01,
                                                  n_pool_games=1000, cv={}))}
    mo = pipeline.decide_matchups("pooled", {"bo3": _val(*NEUTRAL)}, ["bo3"], "TST", others={"bo3": _good(6)})
    html = validate.render_matchups(mo, val)
    assert "Evidence pooled across the other sets" in html and "pooled-better" in html and "(pooled)" in html
