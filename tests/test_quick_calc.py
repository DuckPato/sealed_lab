"""Quick calculation: api.calculate_set forces the four never-useful tests off (no model fitting here)."""
from __future__ import annotations

import json
import sys
import types

from sealed_lab import api
from sealed_lab.model import pipeline

FOUR = dict(manasim="v1", recency="off", attributes="off", extra_draft="off")


def _capture(monkeypatch, tmp_path):
    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path))
    got = {}
    monkeypatch.setattr(pipeline, "calculate", lambda set_code, **kw: got.update(kw))
    fake_corpus = types.ModuleType("sealed_lab.ingest.corpus")
    fake_corpus.build_corpus = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sealed_lab.ingest.corpus", fake_corpus)
    monkeypatch.setattr(api, "_precompute_analysis", lambda sc: None)
    monkeypatch.setattr(api, "set_status", lambda sc: None)
    return got


def test_quick_forces_four_options(monkeypatch, tmp_path):
    got = _capture(monkeypatch, tmp_path)
    api.calculate_set("TST", quick=True, manasim="v2", recency="on", attributes="auto", extra_draft="on", castability="on", matchups="auto")
    assert {k: got[k] for k in FOUR} == FOUR and got["quick"] is True
    assert got["castability"] == "on" and got["matchups"] == "auto" and got["synergy"] == "auto"      # the rest is untouched


def test_not_quick_leaves_options(monkeypatch, tmp_path):
    got = _capture(monkeypatch, tmp_path)
    api.calculate_set("TST", manasim="v2", recency="on", attributes="on", extra_draft="on")
    assert got["quick"] is False
    assert (got["manasim"], got["recency"], got["attributes"], got["extra_draft"]) == ("v2", "on", "on", "on")
    api.calculate_set("TST")
    assert (got["manasim"], got["recency"], got["attributes"], got["extra_draft"]) == ("auto",) * 4


def test_quick_option_values_match_pipeline():
    assert pipeline.QUICK_OPTIONS == FOUR


def test_off_reason_wording():
    assert pipeline.off_reason(False) == "switched off by the user"
    assert pipeline.off_reason(True).startswith("skipped (quick calculation")
    assert pipeline.decide_component("off", "attributes", {}, True, "", "x", quick=True)["reason"] == pipeline.QUICK_OFF_REASON
    assert pipeline.decide_component("off", "attributes", {}, True, "", "x")["reason"] == pipeline.OFF_REASON
    assert pipeline.decide_recency("off", None, {}, {}, quick=True)["reason"] == pipeline.QUICK_OFF_REASON
    per = pipeline.decide_recency_formats("off", {"bo1": None}, {}, {}, quick=True)
    assert pipeline.QUICK_OFF_REASON in per["reason"]
    xd = pipeline.decide_extra_draft("off", {"PickTwo": 5}, 10, None, quick=True)
    assert xd["reason"] == pipeline.QUICK_OFF_REASON and not xd["applied"]


def test_options_travel_as_json():
    json.dumps(dict(set_code="TST", options=dict(quick=True, n_boot=16)))      # calcproc gets the options as JSON kwargs: a bool is fine
